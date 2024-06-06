import tiktoken
from transformers import GPT2Tokenizer, GPT2Model
import torch

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Sample text
text = "This is a sample sentence."

# Tokenize the text using tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)

# Convert tokens to tensor
tokens_tensor = torch.tensor([tokens])
print('Tokens Tensor',tokens_tensor.shape)
# Generate embeddings
with torch.no_grad():
    outputs = model(tokens_tensor)
    #print('GPT-2 shape',outputs.shape)
    last_hidden_states = outputs.last_hidden_state
    print('Last Hidden State',last_hidden_states.shape)

# The embeddings for each token
embeddings = last_hidden_states.squeeze().numpy()
print('Embeddings Shape',embeddings.shape)
# Print embeddings
print(embeddings)