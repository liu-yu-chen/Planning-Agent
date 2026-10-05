"""Resumable low-memory QLoRA SFT for source-grounded literature-review behavior."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train_qlora import chat_ids
from review_metrics import score_answer, aggregate_answers, cited_dois

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SYSTEM = "You are a careful research assistant. Answer only from the supplied literature evidence, and cite factual claims using the exact DOI in square brackets."


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_model(model_path: Path, adapter: Path | None = None, training: bool = True,
               rank: int = 8, alpha: int = 16):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(str(model_path), quantization_config=q,
        device_map={"": 0}, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa")
    base.config.use_cache = not training
    if training:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
    # PEFT upcasts all unquantized weights. The large frozen vocabulary matrices
    # can safely use two-byte storage while normalization/LoRA weights stay fp32.
    base.get_input_embeddings().to(dtype=dtype)
    base.get_output_embeddings().to(dtype=dtype)
    torch.cuda.empty_cache()
    if adapter:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=training)
    elif training:
        model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none", task_type="CAUSAL_LM"))
    else:
        model = base
    model.train(training)
    return model, dtype


def encode_rows(tokenizer, path: Path, max_sequence: int):
    data = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids, prefix = chat_ids(tokenizer, row["prompt"], row["completion"], history=row.get("history"))
        if len(ids) > max_sequence:
            raise ValueError(f"{row['id']}: {len(ids)} > {max_sequence}; target truncation is disabled")
        if not 1 <= len(prefix) < len(ids):
            raise ValueError("Invalid prompt mask")
        data.append({"row": row, "ids": ids, "prefix": len(prefix)})
    return data


def suffix_loss(model, item: dict, dtype):
    ids = torch.tensor([item["ids"]], device="cuda", dtype=torch.long)
    targets = ids[:, item["prefix"]:]
    # Compute logits for assistant targets and their preceding token only.
    # This is equivalent to masking the prompt loss, without allocating prompt
    # positions x the 152k-token vocabulary matrix.
    with torch.autocast(device_type="cuda", dtype=dtype):
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       use_cache=False, logits_to_keep=targets.shape[1] + 1)
        loss = F.cross_entropy(output.logits[:, :-1, :].reshape(-1, output.logits.shape[-1]),
                               targets.reshape(-1))
    return loss, int(targets.numel())


def eval_loss(model, samples: list[dict], dtype, progress_hook=None) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    started = last_print = time.monotonic()
    with torch.no_grad():
        for index, sample in enumerate(samples, 1):
            loss, count = suffix_loss(model, sample, dtype)
            total_loss += float(loss) * count
            total_tokens += count
            if progress_hook:
                elapsed = time.monotonic() - started
                progress_hook(index, len(samples), elapsed)
                if time.monotonic() - last_print >= 30 or index == len(samples):
                    eta = elapsed / index * (len(samples)-index)
                    print(f"Validation loss: {index}/{len(samples)} | elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min", flush=True)
                    last_print = time.monotonic()
    model.train()
    mean = total_loss / max(total_tokens, 1)
    return {"examples": len(samples), "target_tokens": total_tokens,
            "target_loss": mean, "target_perplexity": math.exp(min(mean, 20))}


def generate_answer(model, tokenizer, prompt: str, dtype, max_new_tokens: int = 192, history: list[dict] | None = None, return_details: bool = False):
    messages = [{"role": "system", "content": SYSTEM}] + (history or []) + [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # This SFT teaches concise final answers with an empty reasoning block.
    # The prefill matches those targets; no private chain-of-thought labels exist.
    text += "\n</think>\n\n"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    # The prompt already closes the reasoning block. Some unseen long-context
    # prompts make this small behavior adapter repeat only closing tags. Keep
    # generation in the final-answer channel instead of allowing that loop.
    suppressed = []
    for marker in ('<think>', '</think>'):
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if token_id is not None and tokenizer.convert_ids_to_tokens(token_id) == marker:
            suppressed.append(token_id)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id, use_cache=True,
                                   suppress_tokens=suppressed)
    suffix = generated[0, inputs.input_ids.shape[1]:]
    decoded = tokenizer.decode(suffix, skip_special_tokens=True)
    result = re.sub(r'<think>.*?</think>', '', decoded, flags=re.S).replace('</think>', '').strip()
    if not result:
        raise RuntimeError('The model produced no visible final answer; preserve diagnostics and do not count this generation as successful.')
    elapsed = time.monotonic() - started
    if return_details:
        eos = tokenizer.eos_token_id
        stopped = int(suffix[-1]) == eos if suffix.numel() else False
        return {"answer": result, "generation_seconds": elapsed, "generated_tokens": int(suffix.numel()),
                "reasoning_marker_suppression": suppressed,
                "tokens_per_second": int(suffix.numel())/max(elapsed, 1e-6),
                "generation_limit_reached": int(suffix.numel()) >= max_new_tokens and not stopped,
                "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3}
    return result


def generation_evaluation(model, tokenizer, samples, dtype, max_new_tokens=192, progress_hook=None) -> list[dict]:
    result = []
    started = time.monotonic()
    for sample in samples:
        row = sample["row"]
        details = generate_answer(model, tokenizer, row["prompt"], dtype, history=row.get("history"), max_new_tokens=max_new_tokens, return_details=True)
        answer = details.pop("answer")
        cited = cited_dois(answer)
        allowed = set(row["source_dois"])
        expected = cited_dois(row["completion"])
        result.append({"id": row["id"], "task": row["task"], "domain": row["domain"],
            "question": row["question"], "answer": answer, "reference_answer": row["completion"],
            "source_dois": row["source_dois"], "invalid_cited_dois": sorted(cited - allowed),
            "source_citation_coverage": len(cited & allowed) / len(allowed) if allowed else None,
            "reference_citation_recall": len(cited & expected) / len(expected) if expected else None,
            "reference_requires_citations": bool(expected),
            "metrics": score_answer(answer, row["completion"], row.get("source_evidence") or {d: row["prompt"] for d in row["source_dois"]}) | details,
            "metric_limit": "Citation syntax/coverage does not establish semantic support or review quality."})
        elapsed = time.monotonic() - started
        eta = elapsed / len(result) * (len(samples)-len(result))
        if progress_hook:
            progress_hook(len(result), len(samples), elapsed)
        print(f"Held-out answers: {len(result)}/{len(samples)} | task: {row['task']} | ETA {eta/60:.1f} min", flush=True)
    model.train()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models" / "BASE_HF_CHECKPOINT")
    p.add_argument("--data", type=Path, default=HERE / "review_data" / "train.jsonl")
    p.add_argument("--validation", type=Path, default=HERE / "review_data" / "validation.jsonl")
    p.add_argument("--output", type=Path, default=HERE / "artifacts" / "qlora_review_adapter")
    p.add_argument("--progress", type=Path, default=ROOT / "logs" / "review_qlora_progress.json")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--sequence", type=int, default=640)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--checkpoint-updates", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--initial-adapter", type=Path, help="Continue SFT from a completed adapter with a new optimizer schedule")
    p.add_argument("--max-updates", type=int, default=0)
    p.add_argument("--eval-loss-limit", type=int, default=-1, help="0 evaluates every validation example; -1 preserves the legacy task-balanced subset")
    p.add_argument("--eval-generation-count", type=int, default=0, help="0 preserves the small legacy evaluation; positive counts select domain-balanced examples")
    p.add_argument("--eval-max-new-tokens", type=int, default=192)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")
    torch.manual_seed(20261004)
    random.seed(20261004)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    latest = args.output / "latest_checkpoint.json"
    checkpoint = None
    if args.resume and latest.exists():
        checkpoint = Path(json.loads(latest.read_text(encoding="utf-8"))["path"])
    status = {"status": "loading", "updated_at_utc": utc_now(), "dataset_sha256": dataset_hash}
    write_json(args.progress, status)
    print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = encode_rows(tokenizer, args.data, args.sequence)
    val = encode_rows(tokenizer, args.validation, args.sequence)
    train_dois = {d for x in train for d in x["row"]["source_dois"]}
    val_dois = {d for x in val for d in x["row"]["source_dois"]}
    if train_dois & val_dois:
        raise SystemExit("Training/validation source DOI leakage")
    validation_hash = hashlib.sha256(args.validation.read_bytes()).hexdigest()
    print(f"Training: {len(train)} examples; validation: {len(val)}; max sequence: {args.sequence}; one epoch by default", flush=True)
    print("Loading NF4 base; fp32 normalization/LoRA, two-byte frozen embeddings; active gradient checkpointing...", flush=True)
    model, dtype = load_model(args.model, checkpoint or args.initial_adapter, rank=args.rank, alpha=args.alpha)
    model.print_trainable_parameters()
    parameters = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train) / args.accumulation)
    total_updates = updates_per_epoch * args.epochs
    if args.max_updates:
        total_updates = min(total_updates, args.max_updates)
    warmup = max(1, int(total_updates * 0.08))
    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        fraction = (step - warmup) / max(total_updates - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(fraction, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    seen = updates = 0
    prior_seconds = 0.0
    report_path = args.output / "behavior_evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if checkpoint and report_path.exists() else {}
    if checkpoint:
        state = torch.load(checkpoint / "training_state.pt", map_location="cuda", weights_only=False)
        if state["dataset_sha256"] != dataset_hash:
            raise RuntimeError("Checkpoint dataset hash differs")
        if state.get("validation_sha256", validation_hash) != validation_hash:
            raise RuntimeError("Checkpoint validation data differs")
        if state["accumulation"] != args.accumulation or state["epochs"] != args.epochs:
            raise RuntimeError("Checkpoint training schedule differs")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        seen, updates = state["seen"], state["updates"]
        prior_seconds = state["training_seconds"]
        if "cuda_rng" in state:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        del state
        print(f"Resumed checkpoint: {seen} examples, {updates} optimizer updates", flush=True)
    # A small document-disjoint subset makes the baseline comparison inexpensive.
    val_subset = []
    for task in sorted({s["row"]["task"] for s in val}):
        val_subset += [s for s in val if s["row"]["task"] == task][:2]
    generation_subset = []
    for task in ("reported_finding", "compare_findings", "available_evidence_limit", "insufficient_evidence"):
        choices = [x for x in val if x["row"]["task"] == task]
        if choices:
            generation_subset.append(choices[0])
    if not generation_subset:
        for task in ("finding_quote_evidence", "method_evidence_evidence", "study_area_followup", "review_outline_followup"):
            choices = [x for x in val if x["row"]["task"] == task]
            if choices:
                generation_subset.append(choices[0])
        if not generation_subset:
            for task in sorted({x["row"]["task"] for x in val})[:4]:
                generation_subset.append(next(x for x in val if x["row"]["task"] == task))
    if args.eval_generation_count:
        domains = sorted({s["row"]["domain"] for s in val})
        generation_subset = []
        for index in range(args.eval_generation_count):
            domain = domains[index % len(domains)]
            turn = 1 + index % 2
            choices = [s for s in val if s["row"]["domain"] == domain and s["row"].get("turn", turn) == turn]
            if not choices:
                choices = [s for s in val if s["row"]["domain"] == domain]
            if choices:
                generation_subset.append(choices[(index // len(domains)) % len(choices)])
    if args.eval_loss_limit >= 0:
        val_subset = val[:args.eval_loss_limit] if args.eval_loss_limit else val
    if report.get("baseline_answers") and "baseline_metrics" not in report:
        for result in report["baseline_answers"]:
            result["metrics"] = score_answer(result["answer"], result["reference_answer"], {d: "" for d in result["source_dois"]})
        report["baseline_metrics"] = aggregate_answers(report["baseline_answers"])
    def evaluation_progress(phase, part):
        def update(done, total, elapsed):
            write_json(args.progress, {"status": phase, "evaluation_part": part, "updated_at_utc": utc_now(),
                "evaluation_done": done, "evaluation_total": total, "evaluation_percent": 100*done/max(total, 1),
                "eta_seconds": elapsed/done * (total-done), "training_examples": len(train)})
        return update
    if not checkpoint:
        write_json(args.progress, status | {"status": "baseline_evaluation", "training_examples": len(train)})
        print(f"Baseline held-out evaluation: {len(val_subset)} teacher-forced examples and {len(generation_subset)} answers", flush=True)
        # Check the custom memory-saving objective against the model's masked
        # causal-LM loss before allowing any optimizer update.
        model.eval()
        check = train[0]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            check_ids = torch.tensor([check["ids"]], device="cuda", dtype=torch.long)
            check_labels = check_ids.clone()
            check_labels[:, :check["prefix"]] = -100
            reference_output = model(input_ids=check_ids, attention_mask=torch.ones_like(check_ids),
                                     labels=check_labels, use_cache=False)
            reference = reference_output.loss
            same_logits_loss = F.cross_entropy(
                reference_output.logits[:, check["prefix"] - 1:-1, :].float().reshape(-1, reference_output.logits.shape[-1]),
                check_ids[:, check["prefix"]:].reshape(-1))
            if abs(float(reference) - float(same_logits_loss)) > 1e-5:
                raise RuntimeError("Masked-label alignment differs from the native causal-LM objective")
            optimized, _ = suffix_loss(model, check, dtype)
            difference = abs(float(reference) - float(optimized))
            # Cropping the vocabulary projection changes its low-precision GEMM
            # shape; bf16 rounding can differ slightly even with equal positions.
            if difference > 0.02:
                raise RuntimeError(f"Suffix objective mismatch: {difference}")
            print(f"Masked-loss equivalence verified: difference={difference:.7f}", flush=True)
            del check_ids, check_labels, reference, optimized, reference_output, same_logits_loss
        torch.cuda.empty_cache()
        report["before"] = eval_loss(model, val_subset, dtype, evaluation_progress("baseline_evaluation", "validation_loss"))
        report["baseline_answers"] = generation_evaluation(model, tokenizer, generation_subset, dtype, args.eval_max_new_tokens, evaluation_progress("baseline_evaluation", "generation"))
        report["baseline_metrics"] = aggregate_answers(report["baseline_answers"])
        report["evaluation_scope"] = {"validation_loss_examples": len(val_subset), "validation_total": len(val), "generation_examples": len(generation_subset), "max_new_tokens": args.eval_max_new_tokens,
            "validation_sha256": validation_hash, "generation_ids": [s["row"]["id"] for s in generation_subset]}
        report["note"] = "Document-disjoint but template-derived evaluation. Loss and citation metrics alone do not prove synthesis quality. Full RAG graph/ordinary retrieval evaluation remains a separate task."
        write_json(report_path, report)
        print(f"Baseline target loss: {report['before']['target_loss']:.4f}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    started = time.monotonic()
    start_seen = seen
    last_print = started - 31
    total_examples = len(train) * args.epochs
    stop_requested = False
    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; saving at the next complete optimizer update...", flush=True)
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)

    def save_checkpoint():
        directory = args.output / "checkpoints" / f"update_{updates:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(directory, safe_serialization=True)
        torch.save({"seen": seen, "updates": updates, "dataset_sha256": dataset_hash,
                    "validation_sha256": validation_hash,
                    "accumulation": args.accumulation, "epochs": args.epochs,
                    "training_seconds": prior_seconds + time.monotonic() - started,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "cuda_rng": torch.cuda.get_rng_state()}, directory / "training_state.pt")
        write_json(latest, {"path": str(directory.resolve()), "seen": seen, "updates": updates})
        print(f"Checkpoint saved: {updates}/{total_updates} updates", flush=True)

    try:
        while seen < total_examples and updates < total_updates:
            epoch = seen // len(train)
            order = list(range(len(train)))
            random.Random(20261004 + epoch).shuffle(order)
            position = seen % len(train)
            batch_end = min(position + args.accumulation, len(train))
            micro_count = batch_end - position
            loss_sum = 0.0
            for offset in range(position, batch_end):
                item = train[order[offset]]
                loss, _ = suffix_loss(model, item, dtype)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; refusing to update weights")
                scaler.scale(loss / micro_count).backward()
                loss_value = float(loss.detach())
                loss_sum += loss_value
                seen += 1
                del loss
                elapsed = time.monotonic() - started
                session_seen = max(seen - start_seen, 1)
                eta = elapsed / session_seen * max(total_examples - seen, 0)
                status = {"status": "training", "updated_at_utc": utc_now(),
                    "examples_done": seen, "examples_total": total_examples,
                    "percent": 100 * seen / total_examples, "epoch": epoch + 1,
                    "epochs": args.epochs, "optimizer_updates": updates, "updates_total": total_updates,
                    "loss": loss_value, "elapsed_seconds": round(prior_seconds + elapsed, 1),
                    "eta_seconds": round(eta, 1), "sample_id": item["row"]["id"],
                    "domain": item["row"]["domain"], "task": item["row"]["task"],
                    "cuda_allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 2)}
                write_json(args.progress, status)
                if time.monotonic() - last_print >= 30 or seen == total_examples:
                    filled = min(24, int(24 * seen / total_examples))
                    bar = "#" * filled + "." * (24 - filled)
                    print(f"Review QLoRA: [{bar}] {seen}/{total_examples} ({status['percent']:.1f}%) | update {updates}/{total_updates} | loss {loss_value:.4f} | GPU allocated {status['cuda_allocated_gib']:.2f} GiB | elapsed {status['elapsed_seconds']/60:.1f} min | ETA {eta/60:.1f} min", flush=True)
                    last_print = time.monotonic()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Nonfinite gradients; refusing to save an invalid adapter")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates % args.checkpoint_updates == 0 or seen == total_examples or stop_requested or updates == total_updates:
                save_checkpoint()
            if stop_requested:
                write_json(args.progress, status | {"status": "paused", "optimizer_updates": updates})
                return
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)
        seconds = prior_seconds + time.monotonic() - started
        write_json(args.progress, status | {"status": "final_evaluation", "optimizer_updates": updates, "eta_seconds": None})
        print("Training updates complete. Evaluating the same held-out papers...", flush=True)
        report["after"] = eval_loss(model, val_subset, dtype, evaluation_progress("final_evaluation", "validation_loss"))
        report["adapter_answers"] = generation_evaluation(model, tokenizer, generation_subset, dtype, args.eval_max_new_tokens, evaluation_progress("final_evaluation", "generation"))
        report["adapter_metrics"] = aggregate_answers(report["adapter_answers"])
        report["comparison"] = {"target_loss_change": report["after"]["target_loss"] - report["before"]["target_loss"],
            "metric_change": {key: report["adapter_metrics"]["mean"][key] - value if value is not None and report["adapter_metrics"]["mean"][key] is not None else None
                              for key, value in report["baseline_metrics"]["mean"].items()},
            "scope": "Same held-out sources, prompts, generation budget and decoding before/after SFT."}
        report["created_at_utc"] = utc_now()
        write_json(report_path, report)
        manifest = {"base_model": str(args.model.resolve()), "adapter_type": "QLoRA behavior SFT",
            "dataset_sha256": dataset_hash, "training_examples": len(train), "validation_examples": len(val),
            "validation_sha256": validation_hash,
            "epochs": args.epochs, "optimizer_updates": updates, "training_seconds": round(seconds, 1),
            "rank": args.rank, "alpha": args.alpha, "sequence_limit": args.sequence,
            "compute_dtype": str(dtype), "gradient_accumulation": args.accumulation,
            "initial_adapter": str(args.initial_adapter.resolve()) if args.initial_adapter else None,
            "training_data_note": "Exact source excerpts and authored templates; not expert annotated whole-paper reviews.",
            "evaluation_file": str(report_path), "created_at_utc": utc_now()}
        write_json(args.output / "review_manifest.json", manifest)
        write_json(args.progress, status | {"status": "completed", "optimizer_updates": updates,
            "eta_seconds": 0, "output": str(args.output.resolve()), "evaluation": str(report_path.resolve())})
        print(f"COMPLETE | adapter: {args.output} | before/after held-out target loss: {report['before']['target_loss']:.4f}/{report['after']['target_loss']:.4f}", flush=True)
    except Exception as exc:
        write_json(args.progress, status | {"status": "failed", "error": str(exc), "updated_at_utc": utc_now()})
        raise


if __name__ == "__main__":
    main()
