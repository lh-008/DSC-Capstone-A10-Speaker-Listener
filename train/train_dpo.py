import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefs_jsonl", required=True)
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--output_dir", default="outputs/dpo_smoke")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.prefs_jsonl, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        beta=args.beta,
    )

    if hasattr(dpo_args, "max_length"):
        dpo_args.max_length = args.max_length
    if hasattr(dpo_args, "max_prompt_length"):
        dpo_args.max_prompt_length = min(128, args.max_length // 2)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
