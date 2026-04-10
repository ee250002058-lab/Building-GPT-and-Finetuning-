import torch
import json
from dataset import get_batch, vocab_size
from model import GPT

# use GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

# creating the model and move it to the device
model = GPT(vocab_size)
model = model.to(device)

# AdamW optimizer - this updates the model weights during training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# these lists will store our loss values so we can plot them later
train_losses = []
val_losses   = []

print("Starting training...")

# training loop 
for step in range(20000):

    # get a random batch of training data
    xb, yb = get_batch("train", device=device)

    # forward pass - model makes predictions and calculates how wrong it was
    logits, loss = model(xb, yb)

    # backward pass - figure out how to improve
    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # calculate new gradients

    # gradient clipping - stops gradients from getting too large and breaking training
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update the model weights
    optimizer.step()

    # every 1000 steps, print how the training is going
    if step % 1000 == 0:

        # check loss on validation data to see if we are overfitting
        xv, yv = get_batch("val", device=device)
        _, val_loss = model(xv, yv)

        print(f"Step {step} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

        # save the loss values so we can plot them later
        train_losses.append(round(loss.item(), 4))
        val_losses.append(round(val_loss.item(), 4))

# save the trained model weights to a file
torch.save(model.state_dict(), "model.pth")
print("Model saved")

# save the loss history to a json file so plot_loss.py can read it
with open("loss_log.json", "w") as f:
    json.dump({"train": train_losses, "val": val_losses}, f)
print("Training complete!")
