from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import os


base_model = "microsoft/Phi-3-mini-4k-instruct"
adapter_path = "../models/phi3-process-identification-lora"
merged_path = "../models/phi3-process-identification"

peft_config = PeftConfig.from_pretrained(adapter_path)
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained(merged_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.save_pretrained(merged_path)
