"""QLoRA fine-tuning of a small code model on Spider.

Usage: python train.py            (env vars: MAX_TRAIN=3000, EPOCHS=1, LORA_R=16, USE_WANDB=1)
Output: adapter saved to outputs/adapter
"""
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", 3000))
EPOCHS = float(os.environ.get("EPOCHS", 1))
LORA_R = int(os.environ.get("LORA_R", 16))
MAX_LEN = 768
OUT_DIR = "outputs/adapter"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

rows = [json.loads(line) for line in open("data/train.jsonl")][:MAX_TRAIN]


def encode(r):
    p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
    a = tok(r["query"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:MAX_LEN]
    labels = ([-100] * len(p) + a)[:MAX_LEN]  # loss only on the SQL answer
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


ds = Dataset.from_list(rows).map(encode, remove_columns=list(rows[0].keys()))
ds = ds.filter(lambda x: any(l != -100 for l in x["labels"]))
print("training examples:", len(ds))

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0})
model = prepare_model_for_kbit_training(model)
model = get_peft_model(
    model,
    LoraConfig(
        r=LORA_R,
        lora_alpha=2 * LORA_R,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="outputs/checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    fp16=True,  # T4 GPUs on Kaggle/Colab do not support bf16
    logging_steps=10,
    save_strategy="no",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    report_to="wandb" if os.environ.get("USE_WANDB") else "none",
    run_name=f"text2sql-r{LORA_R}",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
)
trainer.train()
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print("saved adapter to", OUT_DIR)
