import torch
import torch.nn as nn
import torch.nn.functional as F

n_embd = 64   # size of the embedding vector for each character
block_size = 32   # maximum context length
n_heads = 4    # number of attention heads
head_size = 16   # size of each head (n_embd / n_heads = 64 / 4 = 16)
dropout = 0.2  # randomly turn off 20% of neurons during training to avoid overfitting

# ATTENTION HEAD
# This is the core of the transformer.
# Each head learns to figure out: "which past characters are most useful right now?"
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # this is a lower-triangular matrix of 1s
        # it is used to make sure the model cannot look at future characters
        # register_buffer means this will automatically move to GPU if needed
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape   # batch, time (sequence length), channels

        k = self.key(x)    # what information do I have?
        q = self.query(x)  # what information am I looking for?

        # compute attention scores: how much should each token attend to each other?
        scores = q @ k.transpose(-2, -1) * (head_size ** -0.5)

        # mask out future positions (set them to -infinity so softmax gives them 0)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # turn scores into probabilities
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        v   = self.value(x)       # what information do I pass along?
        out = scores @ v          # weighted sum of values
        return out


# MULTI-HEAD ATTENTION
# We run 4 attention heads in parallel and combine their results.
# Each head can focus on different patterns in the text.
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run all heads and concatenate their outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# FEED FORWARD NETWORK
# After attention, each position goes through a small MLP independently.
# This is where the model "thinks" about what it found in attention.
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # expand: 64 -> 256
            nn.GELU(),                         # smooth activation (used in GPT-2)
            nn.Linear(4 * n_embd, n_embd),    # compress back: 256 -> 64
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# TRANSFORMER BLOCK
# One full transformer block = attention + feedforward, with skip connections.
# Skip connections (x + ...) help gradients flow during training.
# LayerNorm stabilizes the values before each sub-layer.
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention   = MultiHeadAttention()
        self.feedforward = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))      # attention + skip connection
        x = x + self.feedforward(self.ln2(x))    # feedforward + skip connection
        return x


# FULL GPT MODEL
class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # token embedding: converts each character id into a 64-dim vector
        self.token_embedding = nn.Embedding(vocab_size, n_embd)

        # position embedding: gives the model a sense of where each character is
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # stack 3 transformer blocks
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()

        # final layer norm
        self.ln = nn.LayerNorm(n_embd)

        # output layer: converts embedding back to vocabulary scores
        self.output = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        # getting token and position embeddings and add them together
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb

        # pass through 3 transformer blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.ln(x)

        # get scores for each character in the vocabulary
        logits = self.output(x)

        # if we have targets, compute the loss
        if targets is None:
            return logits

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    # generate new text character by character
    def generate(self, x, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            # crop to block_size if the context is too long
            x_crop = x[:, -block_size:]

            logits = self(x_crop)

            # only look at the last character's prediction
            logits = logits[:, -1, :]

            # temperature controls randomness: lower = more focused, higher = more random
            logits = logits / temperature

            probs      = F.softmax(logits, dim=-1)
            next_char  = torch.multinomial(probs, num_samples=1)

            # append the predicted character and continue
            x = torch.cat((x, next_char), dim=1)

        return x
