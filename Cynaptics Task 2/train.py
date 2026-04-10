import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset import load_alpaca, format_prompt

# use GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

# load the pretrained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model     = GPT2LMHeadModel.from_pretrained("gpt2")
model     = model.to(device)

tokenizer.pad_token = tokenizer.eos_token

# SPEED SETTINGS — tweak these to control how long it runs

NUM_SAMPLES = 500    # data
MAX_LENGTH  = 64     # sequences 
BATCH_SIZE  = 4       
EPOCHS      = 1      

# load dataset 
data = load_alpaca()
data = list(data)[:NUM_SAMPLES]
print(f"Using {len(data)} training examples")

# AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
model.train()

#list for storing the loss values 
loss_history = []

#training loop
for epoch in range(EPOCHS):
    for i in range(0, len(data), BATCH_SIZE):
        # pick a small batch from dataset
        batch = data[i : i + BATCH_SIZE]

        all_input_ids = []
        all_labels    = []

        for example in batch:
            # full_text = instruction + response (what we feed into the model)
            full_text   = format_prompt(example, include_response=True) 
            prompt_only = format_prompt(example, include_response=False)

            # convert text into tokens 
            full_tokens   = tokenizer.encode(full_text,   truncation=True, max_length=MAX_LENGTH)
            prompt_tokens = tokenizer.encode(prompt_only, truncation=True, max_length=MAX_LENGTH)

            prompt_len = len(prompt_tokens)

            # -100 means ignore this token in loss (only train on response part)
            labels = [-100] * prompt_len + full_tokens[prompt_len:]

            all_input_ids.append(full_tokens)
            all_labels.append(labels)

        # pad all sequences to same length so we find longer ones and pad shorter ones to match it 
        max_len = max(len(x) for x in all_input_ids)

        padded_input_ids = []
        padded_labels    = []
        attention_masks  = []

        for ids, lbls in zip(all_input_ids, all_labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [tokenizer.eos_token_id] * pad_len)
            padded_labels.append(lbls   + [-100]                   * pad_len)
            attention_masks.append([1]  * len(ids) + [0]           * pad_len)

        # convert lists into tensors
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
        labels_tensor    = torch.tensor(padded_labels,    dtype=torch.long).to(device)
        attn_mask_tensor = torch.tensor(attention_masks,  dtype=torch.long).to(device)

        # Forward pass 
        outputs = model(
            input_ids      = input_ids_tensor,
            attention_mask = attn_mask_tensor,
            labels         = labels_tensor
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch + 1} | Step {i} | Loss: {loss.item():.4f}")
            loss_history.append(round(loss.item(), 4))

# save model
model.save_pretrained("my_model")
tokenizer.save_pretrained("my_model")
print("Model saved to my_model/")

import json
with open("loss_log.json", "w") as f:
    json.dump(loss_history, f)
print("Loss log saved to loss_log.json")
