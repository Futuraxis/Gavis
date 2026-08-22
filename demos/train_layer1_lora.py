#!/usr/bin/env python3
"""QLoRA SFT trainer for the Layer 1 local rule translator.

The script fine-tunes a local Hugging Face causal-LM on
``rule_text -> rules_json`` examples, saves the LoRA adapter, then merges
it into a full Hugging Face model at ``models/layer1-rule-llm`` so
``LocalTransformersRuleClient`` can load it directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from layer1_translator.datasets import (
    RuleJsonDataset,
    build_synthetic_examples,
    dump_examples_json,
    load_jsonl_examples,
)

DEFAULT_OUT_DIR = Path("models/layer1-rule-llm")
DEFAULT_ADAPTER_DIR = Path("models/layer1-rule-llm-adapter")


def train(args: argparse.Namespace) -> None:
    """Fine-tune with QLoRA/LoRA and save a merged local model."""
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("缺少训练依赖，请先安装: pip install 'gavis[llm]'") from exc

    examples = build_synthetic_examples()
    if args.data is not None:
        examples.extend(load_jsonl_examples(args.data))
    if not examples:
        raise SystemExit("没有可训练样本")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[name.strip() for name in args.target_modules.split(",") if name.strip()],
    )
    model = get_peft_model(model, lora_config)
    dataset = RuleJsonDataset(examples, tokenizer, args.max_length)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=str(args.adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        bf16=bool(args.bf16 and torch.cuda.is_available()),
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator)
    trainer.train()

    args.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    dump_examples_json(args.adapter_dir / "training_examples.json", examples)

    if args.no_merge:
        print(f"Layer1 LoRA adapter 已保存到: {args.adapter_dir}")
        return

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, args.adapter_dir).merge_and_unload()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir)
    dump_examples_json(args.out_dir / "training_examples.json", examples)
    print(f"Layer1 合并后的完整模型已保存到: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA 训练 Layer1 自然语言规则 -> rules.json 本地模型")
    parser.add_argument("--base-model", required=True, help="本地 Qwen/Llama Hugging Face 基座模型路径")
    parser.add_argument("--data", type=Path, default=None, help="可选 JSONL 训练数据")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="逗号分隔的 LoRA target modules，Qwen2.5/Llama 默认可用",
    )
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-merge", action="store_true", help="只保存 adapter，不合并为完整模型")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
