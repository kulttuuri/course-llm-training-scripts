from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

# Load a smaller subset of English Wikipedia
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]", revision="main", verification_mode="no_checks")
print(f"Dataset size: {len(dataset)} rows")

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Use a pre-trained tokenizer

# Tokenize the text
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets.set_format("torch")

# Configure a small GPT-like transformer
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=128,  # Max sequence length
    n_embd=256,       # Embedding size
    n_layer=4,        # Number of transformer layers
    n_head=4          # Number of attention heads
)

# Create the model
model = GPT2LMHeadModel(config).to("cuda")

# Prepare data loader
data_loader = DataLoader(tokenized_datasets, batch_size=32, shuffle=True)

# Optimizer and loss
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 10
model.train()

for epoch in range(epochs):
    loop = tqdm(data_loader, leave=True)
    for batch in loop:
        # Move data to GPU
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels = batch["input_ids"].to("cuda")
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

# Save the trained model and tokenizer
model.save_pretrained(f"./small_gpt_model_wikitext")
tokenizer.save_pretrained("./small_gpt_model_wikitext")
print("Model saved successfully.")