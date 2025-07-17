# LLMTrainer

This project fine-tunes Meta's LLaMA 3.2-3B using QLoRA (4-bit quantization + LoRA adapters) on multi-turn conversational data from The Tome.

We format conversations into instruction-output pairs, tokenize them, and train a PEFT (LoRA) adapter for the model using the Hugging Face transformers + peft libraries.


# Dataset
Type: Open-ended multi-turn dialogues between "human" and "gpt"

Conversion: Conversations are split into input-output pairs like:

<|begin_of_text|><|start_header_id|>user<|end_header_id|>
[instruction]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
[output]<|eot_id|>

# Model Architecture
Base model: meta-llama/Llama-3.2-3B
LoRA configuration:
  - Target modules: q_proj, k_proj, v_proj, o_proj
  - Rank: 8, Alpha: 32, Dropout: 0.05
  - Quantization: 4-bit (BitsAndBytesConfig)
  - Training framework: Hugging Face Trainer with PEFT integration
  - Logging: Weights & Biases (wandb)
  - Checkpointing: Save every 10k steps, load best at end

# Eval and Train Report
Wandb report: https://wandb.ai/cahmadna-arizona-state-university/Fine-tune%20Llama%203.2%20Tome/reports/Training-Report--VmlldzoxMzYyMTk2Mg

<img width="855" height="651" alt="image" src="https://github.com/user-attachments/assets/ec9d9c6a-8688-401d-8469-9387891abc9f" />

<img width="839" height="535" alt="image" src="https://github.com/user-attachments/assets/4c2b31d6-7b39-4ea1-a6b7-0da344a4eda1" />



Hugging Face model: cameliaariana/STEMAssist

