import torch

# read the text file
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# find all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)     # there are total 65 distinct characters in this dataset

# create a dictionary: character -> number
stoi = {ch: i for i, ch in enumerate(chars)}

# create a dictionary: number -> character
itos = {i: ch for i, ch in enumerate(chars)}

# convert a string to a list of numbers
def encode(s):
    return [stoi[c] for c in s]

# convert a list of numbers back to a string
def decode(l):
    return ''.join([itos[i] for i in l])

# convert the entire text into a tensor of numbers
data = torch.tensor(encode(text), dtype=torch.long)

# split into 90% train and 10% validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

block_size = 32   # how many characters the model sees at once
batch_size = 32   # how many examples we train on at the same time

# this function picks random chunks from the data for training
def get_batch(split, device="cpu"):
    if split == "train":
        data = train_data
    else:
        data = val_data

    # pick random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # x is the input, y is the target (x shifted by 1)
    x = torch.stack([data[i : i + block_size]     for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


