from transformers import GPT2LMHeadModel, AutoTokenizer

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./small_gpt_model_wikitext").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("./small_gpt_model_wikitext")
# Add a new pad_token if not already set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to include the new token

print("Model is ready. Type your prompt and press Enter. Type 'exit' to quit.")

while True:
    # Wait for user input
    text = input("\nEnter your prompt: ")
    
    # Exit condition
    if text.lower() == "exit":
        print("Exiting. Goodbye!")
        break

    # Tokenize the input and compute attention mask
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")

    # Generate text with proper attention mask and pad_token_id
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Explicitly pass attention_mask
        max_length=128,
        pad_token_id=tokenizer.pad_token_id,      # Explicitly pass pad_token_id
        temperature=0.7,                         # Adjust randomness
        top_k=50,                                # Top-k sampling
        do_sample=True,
        repetition_penalty=1.2                   # Penalize repetition
    )

    # Decode and print the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")