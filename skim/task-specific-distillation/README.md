# Finetuning Small Language Model (SLM) for synthetic query generation

In order to finetune SLM for synthetic query generation, we need to perform the following steps in order:

1. Cook up a prompt for your XC (or retrieval) task. You must specify in the prompt what you want to generate given some metadata and a document (or vice-versa). Include as many instructions as you can about the task. Additionally, include some instructions about generating diverse queries since we want to cover as much diverse world knowledge possible. It's very important that the prompt explicitly asks the LLM to only use the world knowledge provided in the metadata to generate synthetic queries (if the LLM uses its own world knowledge, then the finetuned SLM would not have access to this). Prompts used in the paper for LF-ORCAS-800K and LF-WikiTitlesHierarchy-2M datasets are in the `artifacts/prompts/` directory. You may use these as starting points for your task. 

2. Use this prompt to generate synthetic queries for (query, metadata) pairs using a LLM (e.g. GPT4) i.e. we ask the LLM using the prompt that we prepared in step 1 to generate synthetic queries given a query and its associated metadata. These (query, metadata, synthetic queries) are used as training data to finetune a SLM that would be used for large-scale generation. We collected < 50K responses from GPT4 for our tasks for finetuning the SLM, but some tasks may require more LLM responses to be learnt. We used the file `skim/task-specific-distillation/query_gpt_api.py` to collect GPT4 responses for finetuning our SLMs. Our collected GPT4 responses for the datasets used in the paper as well as converted JSONs that can be used for finetuning SLMs using `litgpt` library can be found in `artifacts/llm-responses`

> Note that we use the library `litgpt` in our paper to train SLMs. If you want a general guide on how to use `litgpt`, we recommend starting here at: https://github.com/Lightning-AI/litgpt/blob/main/tutorials/0_to_litgpt.md

3. Convert this dataset into a format that `litgpt` library can accept as training data (these guides may be helpful: https://github.com/Lightning-AI/litgpt/blob/main/tutorials/finetune_lora.md#tune-on-your-dataset, https://github.com/Lightning-AI/litgpt/blob/main/tutorials/prepare_dataset.md#preparing-custom-datasets-from-a-json-file). We provide these files for datasets used in the paper in the dir `artifacts/LLM-responses`. You also need to tokenize your dataset before training the SLM.

4. Finetune your SLM on the training dataset we created in Step 3. You can again use `litgpt` library for this. We provide a modified implementation at `skim/task-specific-distillation/finetune-slm.py`. The sample command used is as follows for running the modified implementation: 
```bash
CUDA_VISIBLE_DEVICES=0 python finetune_slm.py \
--dataset ../../artifacts/LLM-responses/LF-ORCAS-800K \
--model_name meta-llama/Llama-2-7b-hf \
--batch_size 64 \
--micro_batch_size 2 \
--precision bf16-true \
--devices 1 \
--num_epochs 3 \
--learning_rate 3e-4 \
--min_lr 3e-5 \
--warmup_steps 100 \
--eval_interval 25 \
--save_interval 25 \
--lora_query 1 \
--lora_key 1 \
--lora_value 1 \
--lora_projection 1 \
--lora_mlp 1 \
--lora_head 1
```
You need to modify `skim/task-specific-distillation/finetune_slm.py` and copy it into in the `litgpt/finetune` directory (after cloning `litgpt` from: https://github.com/Lightning-AI/litgpt) to make the above command work. Change the `--dataset` argument accordingly before running. Or you may use on the standard LoRA finetuning script that is provided by default in `litgpt` here: https://github.com/Lightning-AI/litgpt/blob/main/litgpt/finetune/lora.py.

5. Merge the LORA weights using the guide here: https://github.com/Lightning-AI/litgpt/blob/main/tutorials/finetune_lora.md#merging-lora-weights-optional.

6. Convert the trained `litgpt` model to huggingface format for inference. Use this guide here: https://github.com/Lightning-AI/litgpt/blob/main/tutorials/convert_lit_models.md

7. Finally, we are ready for large-scale inference. Refer to the dir `skim/step-1` for more instructions on how to do that.

> We provide our finetuned SLMs (in HF format) for datasets used in the paper at: `artifacts/finetuned-slms`.
> We provide the converted training datasets used for training SLMs using `litgpt` at: `artifacts/llm-responses`