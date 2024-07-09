import tiktoken
from transformers import GPT2Tokenizer, GPT2Model
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pickle
import os
from tqdm import tqdm

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
enc = tiktoken.get_encoding("gpt2")

def get_embeddings(sentence):
    # Tokenize the text using tiktoken
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

def get_skills(file_path):
    skills = None
    with open(file_path,'rb') as f:
        skills = pickle.load(f)
    return skills

def load_skills_embedding(file_path):
    skills_embed = None
    if not os.path.exists(file_path):
        return None
    with open(file_path,'rb') as f:
        skills_embed = pickle.load(f)
    return skills_embed

def save_skills_embedding(embedding, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(embedding, file)


def generate_embeddings(skills, save_interval=10):
    all_embeddings = load_skills_embedding('skills_embedding.bin')
    if all_embeddings is None:
        print('Generating embeddings...')
        embeddings = []
        map = {}
        for i, row in tqdm(enumerate(skills)):
            embeddings.append(get_embeddings(row[0]))
            map[row[0]] = i
            if i > 0 and i % save_interval == 0:
                save_skills_embedding({'embedding':embeddings,'map':map},'skills_embedding.bin')

        save_skills_embedding({'embedding':embeddings,'map':map},'skills_embedding.bin')
        all_embeddings = {'embedding':embeddings,'map':map}
        
    else:
        print('Skipping generation, embeddings found in skills_embedding.bin')
        print(f"Found {len(all_embeddings['embedding'])} skills")
        print(f"{all_embeddings['map']}")

    return all_embeddings


    
skills = get_skills("skills.bin")
embeddings = generate_embeddings(skills)
#pca = PCA(n_components = 26)
#print(embeddings)
#reduced_embeddings = pca.fit_transform(embeddings['embedding'])


while True:
    # Define the sentences
    sentence1 = input("Enter sentence 1:  ")
    while sentence1 not in embeddings['map']:
        sentence1 = input('Not found. Re-enter sentence 1:  ')
    sentence2 = input("Enter Sentence 2:  ")
    while sentence2 not in embeddings['map']:
        sentence2 = input('Not found. Re-enter sentence 2:  ')

    
    for i in range(1, 64, 2):
        pca = PCA(n_components = i)
        reduced_embeddings = pca.fit_transform(embeddings['embedding'])
        red_embed1 = reduced_embeddings[embeddings['map'][sentence1]]
        red_embed2 = reduced_embeddings[embeddings['map'][sentence2]]

        # Calculate cosine similarity on reduced dimensions
        cosine_sim = cosine_similarity([red_embed1], [red_embed1])

        print(f"Dims = {i}; Cosine Similarity after PCA: {cosine_sim[0][0]}")
