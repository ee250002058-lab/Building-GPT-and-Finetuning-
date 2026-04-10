import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset import format_prompt   # use the same prompt format as training

# load the fine-tuned model that is saved after training
tokenizer = GPT2Tokenizer.from_pretrained("my_model")
model     = GPT2LMHeadModel.from_pretrained("my_model")
model.eval()

tokenizer.pad_token = tokenizer.eos_token

print("Model loaded!")
print("Type 'exit' to quit.\n")


def generate(user_input):
    # build the prompt in the same format used during training
    # response=False means we leave the model to fill the response
    example = {"instruction": user_input, "input": ""}
    prompt  = format_prompt(example, include_response=False)

    # convert the prompt text to token numbers
    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
        max_length     = 64
    )

    # generate a response
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask     = inputs["attention_mask"],
            max_new_tokens     = 100,       # generate up to 100 new tokens
            do_sample          = True,      # sample randomly (not greedy)
            temperature        = 0.7,       # lower = more focused answers
            top_k              = 50,        # only consider top 50 tokens at each step
            top_p              = 0.90,      # only consider tokens that make up 90% of probability
            repetition_penalty = 1.3,       # penalize repeating the same words
            pad_token_id       = tokenizer.eos_token_id
        )

    # decode the output numbers back to text
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # returning only response 
    if "### Response:" in full_text:
        response = full_text.split("### Response:")[-1].strip()
    else:
        response = full_text.strip()

    return response


# chatbot loop
while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ("exit", "quit"):
        break

    response = generate(user_input)
    print(f"Bot: {response}\n")
