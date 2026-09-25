from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("intfloat/multilingual-e5-base")

print("Model loaded successfully!")

text = "passage: Driving without a valid driving licence is an offence."

embedding = model.encode(
    text,
    normalize_embeddings=True
)

print("Embedding generated!")
print("Dimension:", len(embedding))
print("First 10 values:", embedding[:10])