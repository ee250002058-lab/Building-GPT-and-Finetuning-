import matplotlib          
matplotlib.use("Agg")           #Added bcz of windows DLL security block
import json
import matplotlib.pyplot as plt

# load the loss values that were saved during training
with open("loss_log.json", "r") as f:
    losses = json.load(f)

train_losses = losses["train"]
val_losses   = losses["val"]

# x axis: every 1000 steps
steps = [i * 1000 for i in range(len(train_losses))]

# plot
plt.figure(figsize=(8, 5))

plt.plot(steps, train_losses, label="Train Loss",      color="blue",   marker="o")
plt.plot(steps, val_losses,   label="Validation Loss", color="orange", marker="o")

plt.title("Training and Validation Loss - Task 1")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# save the plot as an image
plt.savefig("loss_curve.png")
print("Loss curve saved as loss_curve.png")

plt.show()
