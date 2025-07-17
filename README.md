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

# Inference
 Safety Alignment Checks
 
 - Harmful Intention
 <img width="1212" height="469" alt="image" src="https://github.com/user-attachments/assets/9af4bc11-6727-4a6f-a83b-8e4f318c90df" />
 
 - Misinformation
<img width="1213" height="526" alt="image" src="https://github.com/user-attachments/assets/cec5c9b9-4973-41fd-8c6e-d7173c09fc9d" />

- Hallucinations
  <img width="1212" height="485" alt="image" src="https://github.com/user-attachments/assets/a6c4fe90-78b2-418b-b751-620bb576c3af" />



Hugging Face model: cameliaariana/STEMAssist

We fine-tuned two models as part of a modular system designed to enhance both content relevance and safety. The first model is specialized in answering STEM-related questions, trained on domain-specific (CoT) datasets to deliver accurate and focused responses in science, technology, engineering, and mathematics. The second model serves as a prompt router and safety filter—it first evaluates whether a prompt is safe (i.e., free from harmful or inappropriate content) and then determines if the prompt falls within the STEM domain. Only prompts that are deemed both safe and STEM-related are forwarded to the specialized model for processing. This approach ensures both the relevance and the safety of generated responses, making the system more robust and reliable.

# Chain of Thought Reasoning
<img width="677" height="538" alt="image" src="https://github.com/user-attachments/assets/ab418382-89af-46d5-b73f-d980ffc23b01" />

# General
<img width="1201" height="500" alt="image" src="https://github.com/user-attachments/assets/caf7045b-7ca7-4a1c-8ef6-c266ccbef828" />



