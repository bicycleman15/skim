"""
Sample usage:

accelerate launch --num_processes 4 --mixed_precision fp16 skim/step-2/gpu_anns.py \
--x-embeddings artifacts/slm-generations/LF-ORCAS-800K/rewrite_doc_embedding.npy \
--y-embeddings artifacts/datasets/LF-ORCAS-800K/trn_doc_embedding.npy \
--num-ngbrs 200

"""

import torch
from accelerate import Accelerator
import numpy as np
import os
import argparse
from tqdm import tqdm
import scipy.sparse as sp
import numba as nb
import torch.distributed as dist

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class EmbeddingsDataset(Dataset):
    def __init__(self, path):
        self.embeddings = np.load(path, mmap_mode="r")
        # self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, 2, 1).reshape(-1, 1)
        print("Loaded embeddings", self.embeddings.shape)
    
    def __len__(self):
        return self.embeddings.shape[0]
        # return 10000

    def __getitem__(self, index):
        return self.embeddings[index]

@nb.njit(cache=True)
def _recall(
    true_labels_indices,
    true_labels_indptr,
    pred_labels_data,
    pred_labels_indices,
    pred_labels_indptr,
    top_k,
):
    fracs = []
    for i in range(len(true_labels_indptr) - 1):
        _true_labels = true_labels_indices[
            true_labels_indptr[i] : true_labels_indptr[i + 1]
        ]
        _data = pred_labels_data[pred_labels_indptr[i] : pred_labels_indptr[i + 1]]
        _indices = pred_labels_indices[
            pred_labels_indptr[i] : pred_labels_indptr[i + 1]
        ]
        top_inds = np.argsort(_data)[::-1][:top_k]
        _pred_labels = _indices[top_inds]
        if len(_true_labels) > 0:
            fracs.append(
                len(set(_pred_labels).intersection(set(_true_labels)))
                / len(_true_labels)
            )
    return np.mean(np.array(fracs, dtype=np.float32))


def recall(true_labels, pred_labels, top_k):
    return _recall(
        true_labels.indices.astype(np.int64),
        true_labels.indptr,
        pred_labels.data,
        pred_labels.indices.astype(np.int64),
        pred_labels.indptr,
        top_k,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Exact Nearest Neighbour Search multi-GPU")
    parser.add_argument("--x-embeddings", type=str, required=True)
    parser.add_argument("--y-embeddings", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-ngbrs", type=int, default=200)
    

    args = parser.parse_args()

    args.embeddings_folder = os.path.dirname(args.x_embeddings)
    # assert args.embeddings_folder == os.path.dirname(args.y_embeddings), "Both x_embeddings and y_embeddings must be in the same folder..."
    print("Embeddings folder is:", args.embeddings_folder)

    accelerate = Accelerator()

    emb_dataset = EmbeddingsDataset(args.x_embeddings)
    num_points = emb_dataset.embeddings.shape[0]
    emb_loader = DataLoader(emb_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=False)
    
    print("Loading y_embeddings from:", args.y_embeddings)
    classifier_embeddings = np.load(args.y_embeddings)
    # take a norm just in case...
    classifier_embeddings = classifier_embeddings / np.linalg.norm(classifier_embeddings, 2, 1).reshape(-1, 1)

    num_labels = classifier_embeddings.shape[0]
    print("[INFO] Total num of labels:", num_labels)
    rank = int(str(accelerate.device).replace("cuda:", ""))
    labels_per_gpu = classifier_embeddings.shape[0] // accelerate.num_processes
    start_label = rank*labels_per_gpu
    end_label = (rank+1)*labels_per_gpu
    if rank == accelerate.num_processes - 1:
        end_label = num_labels
    print(f"label range: ({start_label} : {end_label}) on device: {rank}")
    classifier_embeddings = torch.from_numpy(classifier_embeddings[start_label:end_label]).to(accelerate.device)
    print(classifier_embeddings.shape, "on device:", rank)

    if accelerate.is_main_process:
        all_scores_mat = np.zeros((num_points, args.num_ngbrs), dtype=np.float32)
        all_indices_mat = np.zeros((num_points, args.num_ngbrs), dtype=np.int32)

    start_idx = 0
    for i, test_emb in tqdm(enumerate(emb_loader), disable=(not accelerate.is_main_process), total=len(emb_loader)):    

        gathered_test_emb = test_emb.to(accelerate.device) # (N, 64)

        with torch.no_grad():
            with accelerate.autocast():
                dot_prod = torch.matmul(classifier_embeddings, gathered_test_emb.T).T # (N, L/8)

                top_data, top_inds = torch.topk(dot_prod, k=args.num_ngbrs) # (N, K)
                top_inds += (rank * labels_per_gpu)

                # gather from all GPUs their respective topK's
                top_data_gather = accelerate.gather(top_data.T).T # (N, K*8)
                top_inds_gather = accelerate.gather(top_inds.T).T # (N, K*8)

                # Now do a topk using only K*num_processes top predictions only
                top_data, top_inds_all_temp = torch.topk(top_data_gather, k=args.num_ngbrs, sorted=True, largest=True) # (N, K)
                top_inds = torch.gather(top_inds_gather, 1, top_inds_all_temp)

            if accelerate.is_main_process:
                all_scores, all_indices = top_data.cpu().numpy(), top_inds.cpu().numpy()
                all_scores_mat[start_idx:start_idx + all_scores.shape[0], :] = all_scores
                all_indices_mat[start_idx:start_idx + all_scores.shape[0], :] = all_indices
            
                start_idx += all_scores.shape[0]

    if accelerate.is_main_process: # Dump the neighbours in `x-embeddings` folder
        print("Saving nearest neighbours indices and scores at:", f"{args.embeddings_folder}")
        np.save(f"{args.embeddings_folder}/ngbr_indices_mat.npy", all_indices_mat)
        np.save(f"{args.embeddings_folder}/ngbr_scores_mat.npy", all_scores_mat)
        print("Done...")
