from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import os
import torch
from accelerate import init_empty_weights, dispatch_model
from transformers import AutoConfig

# CONFIG
model_name = "microsoft/Phi-3-mini-4k-instruct"
dataset_path = os.path.join(os.environ["HOME"], "../process_identification_data.jsonl")
output_dir = "../models/phi3-process-identification-lora"
max_length = 4096

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

# Get config first
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

# Prepare quant config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    attn_implementation="eager",
    device_map={"": 0}
)

# Apply quantization flags
model.config.use_cache = False
model.config.is_loaded_in_4bit = True

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules="all-linear"
)
# After LoRA wrap:
model = get_peft_model(model, peft_config)
model._is_quantized = True 
torch.backends.cuda.matmul.allow_tf32 = True

dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print("Logging DOIs from the test set:")
for example in dataset["test"]:
    doi = example.get("doi")
    if doi:
        print("DOI:", doi)


# Apply chat template
def apply_chat_template(example):
    chat = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    tokenized = tokenizer(chat, padding="max_length", truncation=True, max_length=max_length)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_train = dataset["train"].map(apply_chat_template, remove_columns=["messages"])
tokenized_eval = dataset["test"].map(apply_chat_template, remove_columns=["messages"])

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    logging_steps=10,
    save_strategy="epoch",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    learning_rate=5e-6,
    save_total_limit=1,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_num_workers=8
)

# Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train and save
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
