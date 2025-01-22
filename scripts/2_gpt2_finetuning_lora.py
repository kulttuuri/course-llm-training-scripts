from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel

# Load an instruction dataset
dataset = load_dataset("tatsu-lab/alpaca")

# Split the dataset into train and test if "test" doesn't exist
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)

# Preprocess the dataset
def preprocess_function(examples):
    text = [
        f"Instruction: {instruction}\n{input_text}\nResponse: {output}"
        for instruction, input_text, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        )
    ]
    return {"text": text}

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = tokenized_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Load the GPT-2 model and prepare for LoRA fine-tuning
base_model = GPT2LMHeadModel.from_pretrained("gpt2")
#base_model = prepare_model_for_int8_training(base_model)  # Prepare for int8 training (optional)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2", quantization_config=bnb_config)

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank of the adapters
    lora_alpha=32,  # Scaling factor
    target_modules=["c_attn"],  # Apply LoRA to attention layers
    lora_dropout=0.1,  # Dropout for LoRA layers
    bias="none",  # Bias setting ("none", "all", or "lora_only")
    task_type="CAUSAL_LM"  # Task type (e.g., Causal Language Modeling)
)

# Add LoRA adapters to the base model
model = get_peft_model(base_model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_lora_finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    save_steps=500,
    fp16=True  # Enable mixed precision training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

# Train the model with LoRA
trainer.train()

# Save the LoRA model
model.save_pretrained("./gpt2_lora_finetuned")
tokenizer.save_pretrained("./gpt2_lora_finetuned")