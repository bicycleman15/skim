"""
Sample command:

CUDA_VISIBLE_DEVICES=0 python finetune_slm.py \
--dataset ../../artifacts/llm-responses/LF-ORCAS-800K \
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

"""

import os
import sys
import time
import random
import re
import json
import math
import argparse
from tqdm import tqdm
from sys import argv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy

import wandb
import pandas as pd

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    lazy_load,
    num_parameters,
    quantization,
    step_csv_logger,
)

from base_config import (
    RESULTS_DIR, DATA_DIR, CHECKPOINT_DIR
)

# TODO: use dict to autmatically handle this ...
# from scripts.prepare_wiki import generate_prompt
from scripts.prepare_orcas import generate_prompt

def setup(args):
    """Process some parameters and prepare for arguments training
    """
    args.precision = args.precision or get_default_supported_precision(training=True)
    fabric_devices = args.devices
    if fabric_devices > 1:
        if args.strategy == "fsdp":
            if args.quantize:
                raise NotImplementedError(
                    "Quantization is currently not supported for multi-GPU training. "
                    "Please set devices=1 when using the --quantization flag."
                )
            args.strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy={Block},
                state_dict_type="full",
                limit_all_gathers=True,
            )
            print("Using FSDP to train...")
        else:
            print("Using DDP to train...")
            args.strategy = "ddp_spawn"
    else:
        args.strategy = "auto"
    
    # build training directory
    out_dir = Path(
        RESULTS_DIR, 
        args.dataset, 
        f'{args.model_name.split("/")[-2]}_{args.model_name.split("/")[-1]}', 
        datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    )
    os.makedirs(out_dir, exist_ok=True)
    args.out_dir = out_dir
    args.current_time = str(out_dir).split("/")[-1]

    f = open(out_dir / "hparams_log.txt", "w")
    print(f"Setting up train directory: {out_dir}", file=f)
    print(argv, file=f)
    print(args, file=f)
    f.close()

    # logger = step_csv_logger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=fabric_devices, strategy=args.strategy, precision=args.precision)
    fabric.print(f"Setting up train directory: {out_dir}")
    fabric.print(argv)
    fabric.print(args)
    fabric.print(f"Starting training on {fabric_devices} GPU(s)...")
    fabric.launch(
        main, 
        Path(DATA_DIR, args.dataset), 
        Path(CHECKPOINT_DIR, args.model_name), 
        out_dir, 
        args, 
        args.quantize,
    )


def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path, args, quantize: Optional[str] = None):
    check_valid_checkpoint_dir(checkpoint_dir)
    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    # Setup wandb here
    if fabric.global_rank == 0:
        # add more hyperparams to save in wandb
        wandb.init(
            project=args.project,
            config=args,
            name=f"{args.current_time} {args.model_name}"
        )

        wandb.define_metric("loss", summary="min")
        wandb.define_metric("val loss", summary="min")
        

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if not any((args.lora_query, args.lora_key, args.lora_value, args.lora_projection, args.lora_mlp, args.lora_head)):
        fabric.print("Warning: all LoRA layers are disabled!")
    config = Config.from_name(
        name=checkpoint_dir.name,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        to_query=bool(args.lora_query),
        to_key=bool(args.lora_key),
        to_value=bool(args.lora_value),
        to_projection=bool(args.lora_projection),
        to_mlp=bool(args.lora_mlp),
        to_head=bool(args.lora_head),
    )
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=False), quantization(quantize):
        model = GPT(config)
    with lazy_load(checkpoint_path) as checkpoint:
        # strict=False because missing keys due to LoRA weights not contained in state dict
        model.load_state_dict(checkpoint, strict=False)

    mark_only_lora_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if quantize and quantize.startswith("bnb."):
        import bitsandbytes as bnb
        # betas as defined in the LIMA paper
        optimizer = bnb.optim.PagedAdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    else:
        # betas as defined in the LIMA paper
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    model, optimizer = fabric.setup(model, optimizer)

    tokenization_folder = data_dir / config.org / config.name
    fabric.print("Loading data from:", tokenization_folder)
    train_data = torch.load(tokenization_folder / "train.pt")
    val_data = torch.load(tokenization_folder / "test.pt")

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, model, optimizer, train_data, val_data, checkpoint_dir, out_dir, speed_monitor, args)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "lit_model_lora_finetuned.pth"
    save_lora_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
    args: argparse.Namespace
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    max_seq_length, longest_seq_length, longest_seq_ix = get_max_seq_length(train_data, args.override_max_seq_length)
    fabric.print(f"{max_seq_length=}, {longest_seq_length=}, {longest_seq_ix=}")

    # calculate num_iters and micro_batch_size
    gradient_accumulation_iters = args.batch_size // args.micro_batch_size
    assert gradient_accumulation_iters > 0
    fabric.print("-------------------------")
    fabric.print(f"Using per device batch_size of {args.batch_size}, with grad accum of {gradient_accumulation_iters}")
    fabric.print(f"Therefore, total batch_size is {args.batch_size * args.devices}")

    max_iters = (len(train_data) // args.micro_batch_size) // args.devices
    fabric.print(f"Each epoch will use {max_iters} iterations, having ~{len(train_data) // args.batch_size // args.devices} optimizer steps...")
    max_iters = args.num_epochs * max_iters
    fabric.print(f"Therefore, {args.num_epochs} epoch(s) will use {max_iters} iterations and ~{args.num_epochs * len(train_data) // args.batch_size // args.devices} steps...")
    fabric.print("-------------------------")

    # create a train and val metric file to dump the loss in
    if fabric.global_rank == 0: # only have it on the main process
        train_log_file = open(out_dir / "train_log.txt", "w")
        val_log_file = open(out_dir / "val_log.txt", "w")

    # validate(fabric, model, train_data, val_data, tokenizer, longest_seq_length, args)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        mark_only_lora_as_trainable(meta_model)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * args.micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        # this assumes that all samples have a fixed length equal to the longest sequence length
        # which is most likely false during finetuning
        x = torch.randint(0, 1, (args.micro_batch_size, longest_seq_length))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    train_bar = tqdm(range(max_iters), desc="Training", disable=(fabric.global_rank != 0))
    for iter_num in train_bar:

        # if step_count <= args.warmup_steps:
        #     # linear warmup
        #     lr = args.learning_rate * step_count / args.warmup_steps
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr

        # determine and set the learning rate for this iteration
        # this does not take account the step_count, only iterations done
        lr = get_lr(iter_num, args.warmup_steps, args.learning_rate, max_iters, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(
            fabric, train_data, longest_seq_length, args.micro_batch_size, longest_seq_ix if iter_num == 0 else None
        )

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, max_seq_length=max_seq_length, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

        t1 = time.perf_counter()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (iter_num + 1) * args.micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        train_bar.set_postfix_str(
            f"iter {iter_num} step {step_count}: loss {loss.item():.6f}, iter time:"
            f" {(t1 - iter_t0) * 1000:.2f}ms"
            f" lr: {lr:.6f}"
        )
        if iter_num % args.log_interval == 0:
            if fabric.global_rank == 0: # only dump in the main process
                print(f"iter {iter_num} step {step_count}: loss {loss.item():.6f} lr: {lr:.6f}", file=train_log_file, flush=True)
                wandb.log({"iter" : iter_num, "loss" : loss.item(), "lr" : lr, "step" : step_count})

        if not is_accumulating and step_count % args.eval_interval == 0:
            t0 = time.perf_counter()
            val_loss, sample_generations, sample_dicts = validate(fabric, model, train_data, val_data, tokenizer, longest_seq_length, args)
            t1 = time.perf_counter() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            if fabric.global_rank == 0:
                # wandb log
                wandb.log({"iter" : iter_num, "val loss" : val_loss})
                # file log
                print(f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms", file=val_log_file, flush=True)

            ## spit the generations in the log
            fabric.print("Dumping sample generations...")
            if fabric.global_rank == 0: # only dump on the master process
                # first log this in wandb
                # wandb.log({"iter" : iter_num, f"sample_rewrites_{iter_num:06d}" : wandb.Table(data=pd.DataFrame(sample_dicts))})
                # spit these in stdout too
                # print()
                # for i, sample in enumerate(sample_dicts):
                #     print(i, sample["query"], sample["rewrite"], sample["GPT4_rewrite"], \
                #           sample["manual_rewrite"] if "manual_rewrite" in sample else "n/a", sep="  ---  ")
                # print()
                # make a folder if not already present
                dump_folder = out_dir / "sample_generations"
                os.makedirs(dump_folder, exist_ok=True)

                dump_file = open(dump_folder / f"generation-dump-{iter_num:06d}.txt", "w")
                print(f"step {iter_num}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms", file=dump_file)
                print(file=dump_file)
                for sample in sample_generations:
                    print(sample, file=dump_file)
                    print(file=dump_file)
                    print("------------------------------------------", file=dump_file)
                    print(file=dump_file)
                dump_file.close() # close the file
            # wait for all GPUs to sync
            fabric.barrier()
        if not is_accumulating and step_count % args.save_interval == 0:
            checkpoint_path = out_dir / "state_dicts" / f"iter-{iter_num:06d}-ckpt.pth"
            save_lora_checkpoint(fabric, model, checkpoint_path)

    
    if fabric.global_rank == 0: # close the loggers on the main process
        train_log_file.close()
        val_log_file.close()


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, train_data: List[Dict], val_data: List[Dict], tokenizer: Tokenizer, longest_seq_length: int, args
):
    fabric.print("Validating ...")
    model.eval()
    # losses = torch.zeros(args.eval_iters)
    losses = torch.zeros(len(val_data))
    val_bar = tqdm(range(len(val_data)), desc="Validation", disable=(fabric.global_rank != 0))
    for k in val_bar:
        # input_ids, targets = get_batch(fabric, val_data, longest_seq_length, args.micro_batch_size)
        input_ids, targets = get_batch(fabric, val_data, longest_seq_length, 1, None, k)
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
        losses[k] = loss.item()
        val_bar.set_postfix_str(f"val loss : {loss.item():.5f}")
    val_loss = losses.mean()

    sample_queries = list()
    # add some handpicked queries
    # sample_queries += list(json.load(open("/blob/DataAug/Datasets/wikititles-mnar-v1/handpicked_val.json", "r")))
    # produce some 10 examples on test queries
    sample_queries += val_data[:10] # use the first 30+20 queries as val to visualise
    # attach some 10 train queries too
    sample_queries += train_data[:10]

    fabric.print(f"\nGenerating some {len(sample_queries)} samples for visualisation...\n")

    sample_outputs = list()
    sample_output_dicts = list()

    for sample_query in tqdm(sample_queries, desc="Generating samples", disable=(fabric.global_rank != 0)):
        # example = {"query" : sample_query}
        prompt = generate_prompt(sample_query)
        # fabric.print(f"Running for {sample_query}...")
        encoded = tokenizer.encode(prompt, device=fabric.device)
        max_returned_tokens = len(encoded) + 1000
        output = generate(
            model, idx=encoded, max_returned_tokens=max_returned_tokens, max_seq_length=max_returned_tokens, temperature=0.8, eos_id=tokenizer.eos_id
        )
        output = tokenizer.decode(output)

        # extract the rewritten query if possible

        # Attach GPT output in dump
        output += f'\n\n#############\n## GPT4 label:\n'
        output += sample_query["response"]
        output += "\n"

        sample_outputs.append(output)

        # fabric.print(output)
        # fabric.print()
        model.reset_cache()

    model.train()
    return val_loss.item(), sample_outputs, sample_output_dicts


def get_batch(
    fabric: L.Fabric, data: List[Dict], longest_seq_length: int, micro_batch_size: int, longest_seq_ix: Optional[int] = None, specific_idx: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix
    if specific_idx is not None:
        # select this idx as the only one in the batch
        ix[0] = specific_idx

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_max_seq_length(data: List[Dict], override_max_seq_length: Optional[int] = None) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # support easy override at the top of the file
    return (
        override_max_seq_length if isinstance(override_max_seq_length, int) else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )


def save_lora_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Hyperparameter Configuration")

    # Logging and steps
    parser.add_argument("--eval_interval", type=int, default=50, help="Interval for evaluation (steps)")
    parser.add_argument("--save_interval", type=int, default=50, help="Interval for model saving (steps)")
    parser.add_argument("--eval_iters", type=int, default=10, help="Number of iterations for validation on valset")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (iters) for train loss dumping")

    # Training params
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate to use in cosine decay")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size on a GPU")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro batch size on a GPU")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--override_max_seq_length", type=int, default=None, help="Override maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")

    # LoRA specific hyperparams
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA parameter: r")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA parameter: alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--lora_query", type=int, default=1, help="Use LoRA query")
    parser.add_argument("--lora_key", type=int, default=0, help="Use LoRA key")
    parser.add_argument("--lora_value", type=int, default=1, help="Use LoRA value")
    parser.add_argument("--lora_projection", type=int, default=0, help="Use LoRA projection")
    parser.add_argument("--lora_mlp", type=int, default=0, help="Use LoRA MLP")
    parser.add_argument("--lora_head", type=int, default=0, help="Use LoRA head")

    # Dataset and model
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--precision", type=str, default=None, choices=["f16-true", "f16-mixed", "bf16-true", "bf16-mixed"], help="Precision for training")
    parser.add_argument("--quantize", type=str, choices=["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", None], default=None, help="Quantization method")
    parser.add_argument("--strategy", type=str, choices=["auto", "ddp_spawn", "fsdp"], default="auto", help="Training strategy")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")

    # Wandb
    parser.add_argument("--project", type=str, default="ORCAS query generation", help="wandb project name")

    # Parse the command-line arguments
    args = parser.parse_args()
    setup(args)