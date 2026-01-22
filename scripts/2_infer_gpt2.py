import time
from transformers import GPT2LMHeadModel, AutoTokenizer

# Timing model loading
start_time = time.time()

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # Use "gpt2-medium" or "gpt2-large" for larger versions
model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad token ID explicitly if not already set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds.")
print("GPT2 Model is ready for inference. Type 'exit' to quit or press ctrl + c.")

# Interactive loop
while True:
    # Get user input
    input_text = input("\nEnter your prompt: ")
    
    # Exit condition
    if input_text.lower() == "exit":
        print("Exiting. Goodbye!")
        break
    
    # Timing inference
    start_inference_time = time.time()

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")

    # Generate text
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Explicitly pass the attention mask
        max_length=50,          # Maximum length of the output text
        pad_token_id=tokenizer.pad_token_id,  # Ensure correct padding handling
        temperature=0.3,        # Adds randomness to the output
        top_k=50,               # Limits to top-k tokens
        top_p=0.9,              # Nucleus sampling
        do_sample=True,
        repetition_penalty=1.2  # Penalize repeated phrases
    )

    inference_time = time.time() - start_inference_time

    # Decode and print the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    print(f"Inference completed in {inference_time:.2f} seconds.")