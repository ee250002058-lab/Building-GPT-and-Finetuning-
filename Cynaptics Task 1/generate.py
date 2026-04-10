import torch
from model import GPT
from dataset import decode, vocab_size

# use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the trained model
model = GPT(vocab_size)
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

print("Model loaded! Generating text...\n")

# start with a single zero token as the "seed"
# shape is (1, 1) = 1 batch, 1 character
start = torch.zeros((1, 1), dtype=torch.long, device=device)

# generate 1000 new characters
output = model.generate(start, max_new_tokens=1000, temperature=0.8)

# convert the list of numbers back to text and print it
print(decode(output[0].tolist()))
