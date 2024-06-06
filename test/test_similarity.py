import tiktoken
from transformers import GPT2Tokenizer, GPT2Model
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

def get_embeddings(sentence):
    # Tokenize the text using tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(sentence)
    
    # Convert tokens to tensor
    tokens_tensor = torch.tensor([tokens])
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(tokens_tensor)
        last_hidden_states = outputs.last_hidden_state
    
    # Get the mean of the token embeddings (average pooling)
    embeddings = last_hidden_states.mean(dim=1).squeeze().numpy()
    
    return embeddings



while True:
    # Define the sentences
    sentence1 = input("Enter sentence 1:  ")
    sentence2 = input("Enter Sentence 2:  ")
    sentence3 = input("Enter sentence 3:  ")
    sentence4 = input("Enter Sentence 4:  ")
    sentence5 = input("Enter sentence 5:  ")
    sentence6 = input("Enter Sentence 6:  ")
    sentence7 = input("Enter sentence 7:  ")
    sentence8 = input("Enter Sentence 8:  ")
    sentence9 = input("Enter sentence 9:  ")
    sentence10 = input("Enter Sentence 10:  ")

    # Get embeddings for both sentences
    embedding1 = get_embeddings(sentence1)
    embedding2 = get_embeddings(sentence2)
    embedding3 = get_embeddings(sentence3)
    embedding4 = get_embeddings(sentence4)
    embedding5 = get_embeddings(sentence5)
    embedding6 = get_embeddings(sentence6)
    embedding7 = get_embeddings(sentence7)
    embedding8 = get_embeddings(sentence8)
    embedding9 = get_embeddings(sentence9)
    embedding10 = get_embeddings(sentence10)

    # Combine embeddings into a single matrix
    embeddings = [embedding1, embedding2, embedding3, embedding4, embedding5, embedding6, embedding7, embedding8, embedding9, embedding10]

    # Reduce dimensions using PCA
    pca = PCA(n_components=8)  # Adjust n_components as needed
    reduced_embeddings = pca.fit_transform(embeddings)

    # Calculate cosine similarity on reduced dimensions
    cosine_sim = cosine_similarity([reduced_embeddings[0]], [reduced_embeddings[1]])

    print(f"Cosine Similarity after PCA: {cosine_sim[0][0]}")
