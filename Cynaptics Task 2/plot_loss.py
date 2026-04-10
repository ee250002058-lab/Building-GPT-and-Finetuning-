import matplotlib
matplotlib.use("Agg")          #added bcz of windows DLL security error 
import json
import matplotlib.pyplot as plt

# load the loss values that were saved during training
with open("loss_log.json", "r") as f:
    loss_history = json.load(f)

# x axis: every 200 steps, for 2 epochs
steps = [i * 200 for i in range(len(loss_history))]

# plot
plt.figure(figsize=(8, 5))

plt.plot(steps, loss_history, label="Train Loss", color="blue", marker="o")

plt.title("Training Loss - Task 2 (GPT-2 Fine-tuning on Alpaca)")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# save the plot as an image
plt.savefig("loss_curve.png")
print("Loss curve saved as loss_curve.png")

plt.show()
