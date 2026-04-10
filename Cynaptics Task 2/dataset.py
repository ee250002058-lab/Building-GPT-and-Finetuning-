from datasets import load_dataset

# load the alpaca dataset from hugging face 
def load_alpaca():
    dataset = load_dataset("tatsu-lab/alpaca")
    data = dataset["train"].select(range(1500))
    return data


# convert one example from the dataset into a text prompt
def format_prompt(example, include_response=True):
    instruction = example["instruction"].strip()
    input_text  = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    # if there is extra input context, include it
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    # during training we include the answer so the model can learn from it
    # during inference we leave it blank so the model can fill it in
    if include_response:
        return prompt + output
    else:
        return prompt
