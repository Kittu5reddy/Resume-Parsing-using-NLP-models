import os
import pandas as pd
import numpy as np
import faiss
import fitz
from sentence_transformers import SentenceTransformer
import re

index_file = "./FAISS/resume_index.index"
resume_file = "Resume.csv"

if not os.path.exists(index_file):
    print(f"⚠ FAISS index file '{index_file}' not found. Generate it first.")
    exit()

if not os.path.exists(resume_file):
    print(f"⚠ Resume dataset '{resume_file}' not found.")
    exit()

# Load pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(index_file)

# Load resumes dataset
data = pd.read_csv(resume_file)
data["resume_text"] = data["resume_text"].astype(str)  # Ensure text format

# Function to extract text from a PDF using PyMuPDF (Fitz)
def extract_text_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    return text

# Apply text preprocessing
data["resume_text"] = data["resume_text"].apply(preprocess_text)

# Function to match resume with query
def match_resume(query_text, top_k=1):
    query_text = preprocess_text(query_text)

    # Generate embedding for the query
    query_embedding = model.encode(query_text, normalize_embeddings=True).reshape(1, -1)

    # Perform FAISS search
    distances, indices = index.search(query_embedding, k=top_k)

    # Retrieve valid results
    valid_indices = [i for i in indices[0] if i >= 0]
    matches = data.iloc[valid_indices].copy()

    # Convert L2 distance to similarity score
    matches["similarity_score"] = np.exp(-distances[0][:len(valid_indices)])  # Better scaling

    # Sort results
    matches = matches.sort_values(by="similarity_score", ascending=False)

    return matches[["description", "resume_text", "similarity_score"]].to_dict(orient="records")

# Extract text from a sample resume PDF
pdf_text = extract_text_pymupdf("Palvai's Resume-hackerresume.pdf")
print("\n🔹 Extracted Text from Resume:")
print(pdf_text[:1000])  # Print first 1000 chars for preview

# Job Description
job_description = "Looking for a Software Engineer with Python experience in Django, SQL, and cloud platforms."

# Query and match resumes
query = "Software Engineer with Python experience"
matches = match_resume(query)

# Print results
print("\n🔹 Best Matches:")
for match in matches:
    print(f"\n🍌 Matched Resume:\n{match}")
