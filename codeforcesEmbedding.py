import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import load_dataset

ds = load_dataset("open-r1/codeforces", "default",split="train[:1000]") 
n = 1000 #First n elements taken from database

texts = ds  ['description'][:n]
ratings = np.array(ds["rating"][:n])
ids = ds["id"][:n]  
tags = ds["tags"][:n]  # each element is a list of tags

first_tags = [t[0] if len(t) > 0 else "Unknown" for t in tags]

# characters to remove
remove_chars = "$(){}_^\\"

# make translation table (maps each char -> None)
trans_table = str.maketrans('', '', remove_chars)

# if ds is a Python list of strings
texts_clean = [s.translate(trans_table) for s in texts]

def shorten(s, max_words=10):
    return " ".join(s.split()[:max_words]) + "..."

texts_short = [shorten(t) for t in texts_clean]


# Load the Universal Sentence Encoder from TensorFlow Hub
print("Loading USE model...")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Get embeddings
print("Embedding text...")
embeddings = embed(texts_clean).numpy()

# Apply t-SNE
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=40) 
reduced_embeddings = tsne.fit_transform(embeddings)

# Plot
unique_tags = sorted(set(first_tags))
tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}
colors = [tag_to_idx[tag] for tag in first_tags]

# Plot with categorical colors
plt.figure(figsize=(12, 9))
sc = plt.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    c=colors,
    cmap="tab20",  # categorical colormap
    s=40,
    alpha=0.7
)

# Build legend
handles = []
for tag, idx in tag_to_idx.items():
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                              label=tag, markerfacecolor=plt.cm.tab20(idx / len(unique_tags)),
                              markersize=8))
plt.legend(handles=handles, title="First Tag", bbox_to_anchor=(1.05, 1), loc="upper left")

# Annotate points with text
import random
sample_idx = random.sample(range(n), 100)
for i in sample_idx:
    plt.annotate(str(ids[i]), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=7)


plt.title("t-SNE visualization of USE embeddings")
plt.tight_layout()
plt.savefig("tsne_tags.png", bbox_inches="tight")  # legend fully included
plt.show()
