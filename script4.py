from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb, json
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
wandb_token = os.getenv("wandb")

run = wandb.init(
    project='Fine-tune Llama 3.2 Tome',
    job_type="training",
    anonymous="allow",
    id="desert-music-13",
    resume=True
)

run.mark_preempting()

base_model = 'meta-llama/Llama-3.2-3B'

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

file_path = '/scratch/cahmadna/datasets.jsonl'

def stream_jsonl(path):
    with open(path, 'r', encoding='utf-8') as tome:
        for entry in tome:
            yield json.loads(entry)

def convert_conversations_to_instructions(data):
    formatted = []
    for item in data:
        conv = item.get("conversations", [])
        for i in range(0, len(conv) - 1, 2):
            if conv[i]["from"] == "human" and conv[i+1]["from"] == "gpt":
                instruction = conv[i]["value"].strip()
                output = conv[i+1]["value"].strip()
                formatted.append({
                    "instruction": instruction,
                    "output": output
                })
    return formatted

def format_prompt(example):
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        f"{example['output']}<|eot_id|>"
    )

def tokenize(batch):
    prompts = [
        format_prompt({"instruction": inst, "output": out})
        for inst, out in zip(batch["instruction"], batch["output"])
    ]
    tokenized = tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def process_tome_data(path):
    data = []
    for entry in stream_jsonl(path):
        formatted_entries = convert_conversations_to_instructions([entry])
        data.extend(formatted_entries)
    return data

# Load and tokenize data
raw_data = process_tome_data(file_path)
dataset = Dataset.from_list(raw_data)

tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["instruction", "output"],
    num_proc=10,
)

print(f'Dataset len is {len(dataset)}')

# Shuffle and train/val split
shuffled_dataset = tokenized_dataset.shuffle(seed=42).select(range(int(len(dataset) * 0.5)))
split = shuffled_dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
val_dataset = split["test"]

new_model = "llama-3.2-3b-FineTune-Tome_1.0"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

model.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=16,  
    gradient_accumulation_steps=2,  
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=5000,
    eval_strategy="steps",
    eval_steps=10000,
    save_strategy="steps",
    save_steps=10000,
    save_total_limit=2,
    fp16=True,
    report_to="wandb",
    run_name="llama3-finetune-run",
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

steps_per_epoch = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
print(f"Expected steps per epoch: {steps_per_epoch}")

trainer.train(resume_from_checkpoint='/scratch/cahmadna/llama-3.2-3b-FineTune-Tome_1.0/checkpoint-180000')
trainer.evaluate()
trainer.save_model(new_model)
