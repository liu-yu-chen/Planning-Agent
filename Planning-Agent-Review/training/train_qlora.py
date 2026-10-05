"""Low-VRAM QLoRA pilot for the graph/ordinary RAG answer format."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "pilot_train.jsonl"
DEFAULT_OUTPUT = HERE / "artifacts" / "qlora_pilot_adapter"


def chat_ids(tokenizer, prompt: str, completion: str, history: list[dict] | None = None) -> tuple[list[int], list[int]]:
    messages = [
        {"role": "system", "content": "You are a careful research assistant. Answer only from the supplied literature evidence, and cite factual claims using the exact DOI in square brackets."},
    ]
    if history:
        if any(m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str) for m in history):
            raise ValueError("Conversation history must contain user/assistant text turns")
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    prompt_encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
    )
    full_encoded = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": completion}],
        tokenize=True, add_generation_prompt=False, return_dict=True,
    )
    prompt_ids = prompt_encoded["input_ids"]
    full_ids = full_encoded["input_ids"]
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids[0].tolist() if prompt_ids.ndim > 1 else prompt_ids.tolist()
    if isinstance(full_ids, torch.Tensor):
        full_ids = full_ids[0].tolist() if full_ids.ndim > 1 else full_ids.tolist()
    elif full_ids and isinstance(full_ids[0], list):
        full_ids = full_ids[0]
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.tolist()
    elif prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    # Thinking checkpoints can use a different assistant channel in a generation
    # prefix than in a completed assistant turn. Mask the shared prompt tokens and
    # allow only that short assistant-header suffix to contribute to the loss.
    common = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        common += 1
    if common < max(0, len(prompt_ids) - 8):
        raise RuntimeError("Tokenizer chat template diverged before the assistant header; refusing unsafe label masking.")
    return full_ids, prompt_ids[:common]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Hugging Face Transformers checkpoint (not GGUF)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=0, help="Optional hard cap for a smoke test")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to start a slow CPU fallback.")
    # Prefer fused/memory-efficient SDPA kernels when supported by the RTX 3060.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    if not args.data.exists():
        raise SystemExit(f"Training data not found: {args.data}. Run prepare_training_data.py first.")
    examples = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples:
        raise SystemExit("Training dataset is empty.")

    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)
    print(f"Examples: {len(examples)} | epochs: {args.epochs} | max sequence: {args.max_seq_length} | accumulation: {args.gradient_accumulation}", flush=True)
    print("Loading tokenizer and 4-bit NF4 base model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    encoded = []
    for example in examples:
        ids, prompt_ids = chat_ids(tokenizer, example["prompt"], example["completion"])
        if len(ids) > args.max_seq_length:
            raise ValueError(
                f"{example['id']} tokenized to {len(ids)} tokens, exceeding {args.max_seq_length}. "
                "Shorten evidence excerpts rather than silently truncating target answers."
            )
        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
        encoded.append((ids, labels, example["id"]))
    print(f"Tokenized examples: max={max(len(ids) for ids, _, _ in encoded)} tokens", flush=True)

    try:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate, weight_decay=0.01,
        )
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate, weight_decay=0.01,
        )
    except Exception as exc:
        print(f"PagedAdamW8bit unavailable ({exc}); using torch AdamW.", flush=True)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate, weight_decay=0.01,
        )

    update_steps_per_epoch = math.ceil(len(encoded) / args.gradient_accumulation)
    total_steps = update_steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    loss_sum = 0.0
    updates = 0
    micros = 0
    seen_examples = 0
    total_examples = len(encoded) * args.epochs
    stop = False

    for epoch in range(args.epochs):
        order = list(range(len(encoded)))
        for index, sample_index in enumerate(order):
            ids, labels, sample_id = encoded[sample_index]
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            label_ids = torch.tensor([labels], dtype=torch.long, device="cuda")
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids, use_cache=False)
            loss = outputs.loss
            (loss / args.gradient_accumulation).backward()
            loss_value = float(loss.detach().cpu())
            loss_sum += loss_value
            micros += 1
            seen_examples += 1
            del outputs, loss, input_ids, label_ids, attention_mask

            is_update = micros % args.gradient_accumulation == 0 or index == len(order) - 1
            if is_update:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad), 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if args.max_steps > 0 and updates >= args.max_steps:
                    stop = True
            elapsed = max(time.monotonic() - started, 0.001)
            eta = elapsed / seen_examples * max(total_examples - seen_examples, 0)
            vram = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"QLoRA: {seen_examples}/{total_examples} examples ({100*seen_examples/total_examples:.1f}%) "
                f"| epoch {epoch+1}/{args.epochs} | update {updates}/{total_steps} "
                f"| sample {sample_id} | loss {loss_value:.4f} | peak VRAM {vram:.2f} GB "
                f"| elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min",
                flush=True,
            )
            if stop:
                break
        if stop:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    manifest = {
        "base_model": str(args.model), "adapter_type": "QLoRA",
        "quantization": "4-bit NF4 with double quantization; fp16 compute",
        "lora": {"rank": args.rank, "alpha": args.alpha, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "max_sequence_length": args.max_seq_length, "examples": len(examples),
        "epochs_requested": args.epochs, "optimizer_updates": updates,
        "gradient_accumulation": args.gradient_accumulation,
        "training_seconds": round(time.monotonic() - started, 1),
        "max_cuda_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "dataset_note": "Small manually authored paired pilot derived from the existing eight-question benchmark; not a held-out evaluation set and not a release-ready adapter.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "pilot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved pilot adapter to {args.output}", flush=True)


if __name__ == "__main__":
    main()
