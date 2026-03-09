#!/usr/bin/env python3
"""T-102: Merge LoRA adapter into base model for GGUF conversion."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--base-model", default="models/Qwen3-14B")
    p.add_argument("--adapter-path", default="output/adapter")
    p.add_argument("--output-dir", default="output/merged_model")
    args = p.parse_args()

    print(f"Loading base model (FP16): {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
    )

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    AutoTokenizer.from_pretrained(args.adapter_path).save_pretrained(args.output_dir)

    print(f"Merge complete ✓")

if __name__ == "__main__":
    main()
