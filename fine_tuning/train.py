#!/usr/bin/env python3
"""T-101: QLoRA fine-tuning of Qwen3-14B for 3GPP Root Cause Analysis."""
import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen3-14B with QLoRA")
    p.add_argument("--dataset", default="data/training_data.json")
    p.add_argument("--base-model", default="models/Qwen3-14B")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq-length", type=int, default=2048)
    return p.parse_args()

def main():
    args = parse_args()

    # 1. Load and validate dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    assert len(dataset) == 1300, f"Expected 1300 examples, got {len(dataset)}"
    assert "text" in dataset.column_names, f"Expected 'text' column, got {dataset.column_names}"
    print(f"Dataset: {len(dataset)} examples ✓")

    # 2. Load base model in 4-bit NF4
    print(f"Loading base model: {args.base_model} (4-bit NF4)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on {model.device} ✓")

    # 3. Attach QLoRA adapters
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA attached: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%) ✓")

    # 4. Train
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="no",
        max_grad_norm=0.3,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    print(f"Training: {args.epochs} epoch(s), lr={args.lr}, rank={args.rank}, alpha={args.alpha}")
    trainer.train()

    # 5. Save adapter
    adapter_path = f"{args.output_dir}/adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path} ✓")

    # Summary
    logs = trainer.state.log_history
    losses = [l["loss"] for l in logs if "loss" in l]
    if losses:
        print(f"\nTraining Summary:")
        print(f"  Initial loss: {losses[0]:.3f}")
        print(f"  Final loss:   {losses[-1]:.3f}")
        print(f"  Steps:        {trainer.state.global_step}")
        print(f"  Duration:     {trainer.state.log_history[-1].get('train_runtime', 0):.0f}s")

if __name__ == "__main__":
    main()
