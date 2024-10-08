"""
Sample command: (modify accordingly)

CUDA_VISIBLE_DEVICES=0 python large_scale_inference.py \
--model_name meta-llama/Llama-2-7b-hf \
--model_path /path/to/finetuned_slm/lit_model_merged_lora.bin \
--batch_size 8 \
--torch_dtype torch.float16 \
--title_content_path /path/to/train_queries/and/its/metadata.json \
--query_rewrites_output_path /path/to/dump/synthetic/queries \
--start_index 0 \
--end_index 100 \
--job_index 123

"""

import os
import sys
import numpy as np
import torch
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GPTNeoXForCausalLM, LlamaForCausalLM
import re
from collections import OrderedDict
# from optimum.bettertransformer import BetterTransformer
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# Below assumes we use LF-ORCAS-800K dataset
# Modify below accordingly for LF-WikiTitlesHierarchy-2M
def generate_prompt(example):
    doc_url = example["url"]
    # doc_title = example["doc_title"]
    relevant_queries = example["relevant_queries"]
    context = example["pruned_text"]

    prompt = f"### URL\n{doc_url}\n\n"

    relevant_queries = "\n".join(relevant_queries)
    prompt += f"#### Relevant Queries\n{relevant_queries}\n\n"
    prompt += f"### Webpage text for URL begins\n{context}\n### Webpage text for URL ends"
    prompt += "\n\n### Task Output\n"
    return prompt

def get_percentiles(a):
    perc = [10, 25, 50, 75, 80, 90, 95, 99, 99.9, 99.99]
    for x in perc:
        print(f"{x} th percentile:", np.percentile(a, x).round(2))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hyperparameter Configuration")

    # Logging and steps
    parser.add_argument("--model_name", type=str, required=True, help="model name to infer on")
    parser.add_argument("--model_path", type=str, required=True, help="model saved path in HF format")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size to use for inference")
    parser.add_argument("--torch_dtype", type=str, default="torch.bfloat16", help="torch dtype to use to load the model")

    # parser.add_argument("--limit_prompt_tokens", type=int, default=1500, help="Ignore prompts having more than X tokens")
    parser.add_argument("--token_batch_size", type=int, default=128, help="batch size to use for tokenization")

    parser.add_argument("--title_content_path", required=True, help="tsv file containing query and its corresponding metadata")
    parser.add_argument("--query_rewrites_output_path", required=True, help="tsv file containing query and its synthetic query")

    parser.add_argument("--start_index", type=int, default=0, help="Train points to consider (start index)")
    parser.add_argument("--end_index", type=int, default=1000, help="Train points to consider (end index)")
    parser.add_argument("--job_index", type=int, default=123, help="Job index for tracking runs")
    args = parser.parse_args()
    print(args)

    # Set device using CUDA_VISIBLE_DEVICES=X
    # Here, we always move it to device 0
    DEVICE = "cuda:0"

    print("########## Using precision:", args.torch_dtype)

    config = AutoConfig.from_pretrained(args.model_name)
    print("Loading state_dict from:", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, torch_dtype=eval(args.torch_dtype))
    state_dict = torch.load(args.model_path)
    print(model.load_state_dict(state_dict))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Initial weights dtype:", model.dtype)
    model.to(device=DEVICE, dtype=eval(args.torch_dtype))
    print("Final weights dtype:", model.dtype)
    model.eval()

    # read the file first, and create prompts
    train_data = json.load(open(args.title_content_path, "r"))[args.start_index : args.end_index]
    train_indices = list(range(args.start_index, args.end_index))

    ################################################# 

    LIMIT_CONTENT = 7500 # Limit to 1500 characters
    pruned_contents = list()
    for x in tqdm(train_data):
        pruned_text = (str(x["doc_title"]) + " " + str(x["doc_body"]))[:LIMIT_CONTENT]
        pruned_contents.append(pruned_text)

    char_lens = np.array([len(x) for x in pruned_contents])
    get_percentiles(char_lens)

    temp_input_ids_list = list()
    temp_attention_mask_list = list()
    BATCH_SIZE_TOKEN = args.token_batch_size
    for start_idx in tqdm(range(0, len(pruned_contents), BATCH_SIZE_TOKEN), desc="Tokenizing"):
        end_idx = start_idx + BATCH_SIZE_TOKEN
        prompt_tokenized = tokenizer(pruned_contents[start_idx : end_idx], return_tensors="pt", padding=True)
        tok_lens = [torch.sum(x > 0) for x in prompt_tokenized["attention_mask"]]
        for i in range(len(tok_lens)):
            tok_len = tok_lens[i]
            temp_input_ids_list.append(prompt_tokenized["input_ids"][i, -tok_len:])
            temp_attention_mask_list.append(prompt_tokenized["attention_mask"][i, -tok_len:])

    LIMIT_CONTENT_TOKEN = 750 # Use more for LF-WikiTitlesHierarchy-2M (~1000 tokens)
    temp_input_ids_list = [x[:LIMIT_CONTENT_TOKEN] for x in temp_input_ids_list]
    temp_attention_mask_list = [x[:LIMIT_CONTENT_TOKEN] for x in temp_attention_mask_list]

    for i in tqdm(range(len(train_data))):
        train_data[i]["pruned_text"] = tokenizer.decode(temp_input_ids_list[i][1:]) ## remove the <s> token

    #################################################

    prompts = [generate_prompt(x) for x in train_data]
    print(f"Generating rewrites for {len(train_data)} queries...")

    prompt_input_ids_list = list()
    prompt_attention_mask_list = list()

    BATCH_SIZE_TOKEN = args.token_batch_size
    for start_idx in tqdm(range(0, len(prompts), BATCH_SIZE_TOKEN), desc="Tokenizing"):
        end_idx = start_idx + BATCH_SIZE_TOKEN
        prompt_tokenized = tokenizer(prompts[start_idx : end_idx], return_tensors="pt", padding=True)
        tok_lens = [torch.sum(x > 0) for x in prompt_tokenized["attention_mask"]]
        for i in range(len(tok_lens)):
            tok_len = tok_lens[i]
            prompt_input_ids_list.append(prompt_tokenized["input_ids"][i, -tok_len:])
            prompt_attention_mask_list.append(prompt_tokenized["attention_mask"][i, -tok_len:])

    tokenized_lens = np.array([len(x) for x in prompt_input_ids_list])
    get_percentiles(tokenized_lens)
    print("Num prompts:", tokenized_lens.shape)

    # LIMIT_TOKENS = args.limit_prompt_tokens
    sorted_order = np.argsort(tokenized_lens)
    sorted_order = sorted_order[::-1] # do generation in a reverse order so that OOMs can occur early !

    # keep query rewrites here
    query_rewrite_list = list()

    for start in tqdm(range(0, sorted_order.shape[0], args.batch_size), desc="Generating rewrites"):

        # prompts_indices = range(start, start + args.batch_size)
        prompts_indices = sorted_order[start : start + args.batch_size] # process in order of increasing tok lens
        
        prompt_input_ids = [prompt_input_ids_list[x] for x in prompts_indices]
        prompt_attention_masks = [prompt_attention_mask_list[x] for x in prompts_indices]

        input_lens = [len(s) for s in prompt_input_ids]
        max_len = max(input_lens)

        def pad_left(x, pad_id):
            # pad left based on the longest sequence in the batch
            n = max_len - len(x)
            return torch.cat((torch.full((n,), pad_id, dtype=x.dtype), x))

        prompt_input_ids = torch.stack([pad_left(x, pad_id=tokenizer.pad_token_id) for x in prompt_input_ids])
        prompt_attention_masks = torch.stack([pad_left(x, pad_id=0) for x in prompt_attention_masks])

        prompt_input_ids = prompt_input_ids.to(DEVICE)
        prompt_attention_masks = prompt_attention_masks.to(DEVICE)


        with torch.inference_mode():

            # force to use fast SDPA kernel
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            ):
                output_token_ids = model.generate(
                    inputs=prompt_input_ids,
                    attention_mask=prompt_attention_masks,
                    max_new_tokens=250, # change for LF-WikiTitlesHierarchy-2M (~400-500 tokens)
                    do_sample=True,
                    top_k=200,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,
                    # remove_invalid_values=True # protect aganist failing generate due to nan, inf values
                )

            # only keep the output tokens
            output_token_ids = [out[in_len + (max_len - in_len):] for out, in_len in zip(output_token_ids, input_lens)]
            
            for x, idx in zip(output_token_ids, prompts_indices):
                x = tokenizer.decode(x, skip_special_tokens=True)
                # TODO: can we speed this append up ?
                query_rewrite_list.append((idx, train_indices[idx], x))

    # sort the rewrites first
    query_rewrite_list.sort(key=lambda x : x[0])

    # dump everything in sample_outputs
    dump_file_name = f"orcas_rewrites_jobidx_{args.job_index:05d}_trainidx_{args.start_index:07d}_{args.end_index:07d}.pth"
    final_save_path = os.path.join(args.query_rewrites_output_path, dump_file_name)

    print("Dumping rewrites in a file at:", final_save_path)
    torch.save(query_rewrite_list, final_save_path)
    print("All done...")
