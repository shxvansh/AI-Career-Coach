import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# Configuration
# Use a smaller model for M1 Mac (0.5B params)
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
new_model_name = "finetuning/resume-expert-lora"
dataset_path = "finetuning/dataset.jsonl"

def format_instruction(sample):
    """Format the training data into a prompt."""
    return f"""### Instruction:
Analyze the resume gap based on the job description and provide specific bullet points in LaTeX format.

### Job Description:
{sample['job_description_context']}

### Resume Context:
{sample['resume_context']}

### Output Analysis:
{sample['output_analysis']}
"""

def train():
    print(f"Loading model: {base_model_name}")
    
    # M1 Mac Optimization: Load in float16 if MPS (Metal Performance Shaders) is available
    # But Qwen-0.5B is small enough we can just load it standard for stability if needed.
    # We'll trust AutoModel to handle it, or force float32 if you run into MPS ops errors.
    # For now, let's stick to standard loading. 
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading dataset from {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Run generate_data.py first.")
        return

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all linear layers for Qwen
    )

    sft_config = SFTConfig(
        output_dir="./finetuning/results",
        num_train_epochs=3, # Increased to 3 for better learning on small dataset
        per_device_train_batch_size=2, # Small batch for M1
        gradient_accumulation_steps=4,
        optim="adamw_torch", # Standard optimizer, avoiding bitsandbytes dependency issues
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False, # M1 usually doesn't love bf16 depending on the chip, float16 or 32 is safer
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        use_mps_device=torch.backends.mps.is_available(), # deprecated but still supported in this version
        max_length=1024, # Limit context length for RAM
        packing=False,
    )

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
        formatting_func=format_instruction,
    )

    trainer.train()
    
    print(f"Saving model to {new_model_name}")
    trainer.model.save_pretrained(new_model_name)
    tokenizer.save_pretrained(new_model_name)

if __name__ == "__main__":
    train()

