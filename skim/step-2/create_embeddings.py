"""
Sample Usage: 

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision fp16 skim/step-1/create_embeddings.py \
--raw_text_file_path artifacts/step-1/slm-large-scale-inference/orcas/raw_rewrites.txt \
--prefix rewrite_doc \
--tf sentence-transformers/msmarco-distilbert-base-v4 \
--tokenized_path artifacts/step-1/slm-large-scale-inference/orcas/bert-base-uncased-32 \
--max_len 32 \
--trained_state_dict artifacts/baselines/NGAME/orcas/state_dict.pt

"""
from typing import OrderedDict
import math
import numpy as np
import os
import time
from tqdm import tqdm
import functools
import argparse
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import scipy.sparse as sp
from transformers import AutoModel
import torch.nn.functional as F
import sentence_transformers

from contextlib import contextmanager

@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def timeit(func): 
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer

class STransformerInputLayer(nn.Module):
    """
    Sentence transformer
    """
    def __init__(self, transformer='roberta-base'):
        super(STransformerInputLayer, self).__init__()
        if isinstance(transformer, str):
            self.transformer = sentence_transformers.SentenceTransformer(transformer)
        else:
            self.transformer = transformer

    def forward(self, data):
        sentence_embedding = self.transformer(data)['sentence_embedding']
        return sentence_embedding

class CustomEncoder(nn.Module):
    """
    Encoder layer with Sentence transformer and an optional projection layer

    * projection layer is applied after reduction and normalization
    """
    def __init__(self, encoder_name, transform_dim):
        super(CustomEncoder, self).__init__()
        self.encoder = STransformerInputLayer(sentence_transformers.SentenceTransformer(encoder_name))
        self.transform_dim = transform_dim

    def forward(self, input_ids, attention_mask):
        return self.encoder({'input_ids': input_ids, 'attention_mask': attention_mask})

    @property
    def repr_dims(self):
        return  768

class SiameseNetwork(nn.Module):
    """
    A network class to support Siamese style training
    * specialized for sentence-bert or hugging face
    * hard-coded to use a joint encoder

    """
    def __init__(self, encoder_name, transform_dim, device, normalize_repr):
        super(SiameseNetwork, self).__init__()
        self.padding_idx = 0
        self.encoder = CustomEncoder(encoder_name, transform_dim)
        self.device = device
        self.normalize_repr = normalize_repr
        
    def encode(self, doc_input_ids, doc_attention_mask):
        rep = self.encoder(doc_input_ids.to(self.device), doc_attention_mask.to(self.device))
        if self.normalize_repr:
            rep = F.normalize(rep)
        return rep

    def encode_document(self, doc_input_ids, doc_attention_mask, *args):
        rep = self.encoder(doc_input_ids.to(self.device), doc_attention_mask.to(self.device))
        if self.normalize_repr:
            rep = F.normalize(rep)
        return rep

    def encode_label(self, lbl_input_ids, lbl_attention_mask):
        rep = self.encoder(lbl_input_ids.to(self.device), lbl_attention_mask.to(self.device))
        if self.normalize_repr:
            rep = F.normalize(rep)
        return rep
    
    def forward(self, doc_input_ids, doc_attention_mask, lbl_input_ids, lbl_attention_mask):
        if(doc_input_ids is None):
            return self.encode_label(lbl_input_ids, lbl_attention_mask)
        elif(lbl_input_ids is None):
            return self.encode_document(doc_input_ids, doc_attention_mask)
        doc_embeddings = self.encode_document(doc_input_ids, doc_attention_mask)
        label_embeddings = self.encode_label(lbl_input_ids, lbl_attention_mask)
        return doc_embeddings, label_embeddings

    @property
    def repr_dims(self):
        return self.encoder.repr_dims

def collate_fn_embedding(batch):
    batch_data = {}
    batch_size = len(batch)
    batch_data['batch_size'] = batch_size
    
    batch_data['indices'] = torch.LongTensor([item[0] for item in batch])
    batch_data['ii'] = torch.vstack([item[1] for item in batch])
    batch_data['am'] = torch.vstack([item[2] for item in batch])

    return batch_data

class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, tokenization_folder, num_Z, max_len, prefix_str):
        self.num_Z = num_Z
        self.input_ids = np.memmap(f"{tokenization_folder}/{prefix_str}_input_ids.dat", mode='r', shape=(num_Z, max_len), dtype=np.int64)
        self.attention_mask = np.memmap(f"{tokenization_folder}/{prefix_str}_attention_mask.dat", mode='r', shape=(num_Z, max_len), dtype=np.int64)
    
    def __getitem__(self, index):
        return (index, torch.from_numpy(self.input_ids[index]), torch.from_numpy(self.attention_mask[index]))
    
    def __len__(self):
        return self.num_Z


@timeit
def get_embeddings(model, tokenization_folder, num_Z, max_len, prefix_str, bsz=512):
    data_loader = torch.utils.data.DataLoader(
        dataset=LabelDataset(tokenization_folder, num_Z, max_len, prefix_str),
        batch_size=bsz,
        collate_fn=collate_fn_embedding,
        shuffle=False
        )
    data_loader = accelerator.prepare(data_loader)

    pbar = tqdm(enumerate(data_loader, 0), disable=not accelerator.is_main_process, total=len(data_loader))
    with evaluating(model), torch.no_grad():
        for i, batch in pbar:
            output = model(batch["ii"], batch["am"], None, None)
            output = accelerator.gather(output)
            indices = accelerator.gather(batch["indices"])
            if accelerator.is_main_process:
                out = output.cpu().numpy()
                if(i == 0):
                    embeddings = np.zeros((num_Z, out.shape[1]), dtype=np.float32)
                embeddings[indices.cpu().numpy()] = out 
    if accelerator.is_main_process:
        return embeddings
    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_state_dict", type=str, required=True, help="Model path containing saved NGAME weights")
    parser.add_argument("--raw_text_file_path", type=str, required=True, help="This points to either trn_X.txt, Y.txt, raw_rewrites.txt file")
    parser.add_argument("--prefix_str", type=str, required=True, help="change this according to the raw_text_file_path above, options are [trn_doc, lbl, rewrite_doc]")
    parser.add_argument("--tokenized_path", type=str, required=True, help="Path to the tokenized files for raw_text_file_path")
    parser.add_argument("--tf", type=str, required=True, help="which encoder to use")
    parser.add_argument("--max-len", type=int, required=True, help="Max len used in tokenization. Used to get the tokenization folder")
    args = parser.parse_args()

    ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_handler])

    print("Using Encoder: {}".format(args.tf))
    model = SiameseNetwork(args.tf, -1, 'cuda', True)

    state_dict_path = args.trained_state_dict
    folder_path = os.path.dirname(state_dict_path)
    
    print("Loading state dict from:", state_dict_path)
    state_dict = torch.load(state_dict_path)
    print(model.load_state_dict(state_dict))

    device = accelerator.device
    model.to(device)
    model = accelerator.prepare(model)

    num_rewrites = len([x.strip() for x in open(args.raw_text_file_path, "r").readlines()])
    embs = get_embeddings(model, args.tokenized_path, num_rewrites, args.max_len, args.prefix_str)

    if accelerator.is_main_process:
        print(embs.shape)
        print("Saving embeddings at:", folder_path)
        np.save(os.path.join(folder_path, f"{args.prefix_str}_embeddings.npy"), embs)
        print("Done...")
