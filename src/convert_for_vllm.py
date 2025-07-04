#!/usr/bin/env python3
"""
Convert LoRA fine-tuned model to vLLM compatible format.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

def convert_lora_to_vllm():
    """Convert LoRA model to base model format for vLLM."""
    
    # Load the base model
    base_model_name = "google/flan-t5-small"
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load LoRA adapter
    lora_model = PeftModel.from_pretrained(base_model, "fine-tuned")
    
    # Merge LoRA weights with base model
    merged_model = lora_model.merge_and_unload()
    
    # Save the merged model
    output_dir = "vllm-model"
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Model converted and saved to {output_dir}")
    print("This model can now be used with vLLM")

if __name__ == "__main__":
    convert_lora_to_vllm() 