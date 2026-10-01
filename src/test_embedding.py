from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Load embedding model
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ============================================================
# 2. Example sentences
# ============================================================

sentences = [
    "LHM uses 3D Gaussian Splatting for human reconstruction.",
    "LHM represents a human using 3D Gaussians.",
    "Neo4j is a graph database.",
]


# ============================================================
# 3. Convert text into vectors
# ============================================================

embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)


# ============================================================
# 4. Inspect embeddings
# ============================================================

print("\nNumber of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

print("\nFirst 10 values of sentence 1:")
print(embeddings[0][:10])


# ============================================================
# 5. Calculate cosine similarity
# ============================================================

similarity_1_2 = embeddings[0] @ embeddings[1]
similarity_1_3 = embeddings[0] @ embeddings[2]


print("\nCosine similarities:")

print(
    "Sentence 1 vs Sentence 2:",
    float(similarity_1_2)
)

print(
    "Sentence 1 vs Sentence 3:",
    float(similarity_1_3)
)