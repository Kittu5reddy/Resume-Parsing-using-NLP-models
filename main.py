import os
import pandas as pd
import numpy as np
import faiss
import fitz
from sentence_transformers import SentenceTransformer
import re
#ml logic
index_file = "./FAISS/resume_index.index"
resume_file = "Resume.csv"



# Load pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

import spacy

nlp = spacy.load("en_core_web_sm")

def extract_college_ner(text):
    doc = nlp(text)  # Process text with spaCy
    for ent in doc.ents:
        if ent.label_ == "ORG":  # "ORG" represents organizations (colleges, universities, etc.)
            return ent.text  # Return the first found college name
    return None 

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

    l=matches[["description", "resume_text", "similarity_score"]].to_dict(orient="records")
    # Get the match with the highest similarity score
    return max(l, key=lambda x: x["similarity_score"])["similarity_score"]



import spacy
import fitz
import re
import json
from sentence_transformers import SentenceTransformer, util

import spacy
import fitz
import re
import json
from sentence_transformers import SentenceTransformer, util

import spacy
import fitz
import re
import json
from sentence_transformers import SentenceTransformer, util

class ResumeCertificationValidator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # For better matching

        # Section Headers for Resume Parsing
        self.section_patterns = {
            'projects': r'(?i)(projects?|work\s+experience|technical\s+experience)\s*:?',
            'skills': r'(?i)(technical\s+)?skills\s*:?',
            'certifications': r'(?i)certifications?\s*:?'
        }

        # Updated Certification Database
        self.cert_database = {
    "Programming in Java - NPTEL": {
        "skills": ["java", "object-oriented programming", "spring boot"],
        "patterns": [r"(?i)programming\s+in\s+java\s+-?\s*nptel", r"(?i)nptel\s+java\s+certification"]
    },
    "Introduction to Cybersecurity": {
        "skills": ["cybersecurity", "network security", "ethical hacking"],
        "patterns": [r"(?i)introduction\s+to\s+cybersecurity"]
    },
    "Cybersecurity Essentials": {
        "skills": ["cybersecurity", "risk management", "encryption"],
        "patterns": [r"(?i)cybersecurity\s+essentials"]
    },
    "Smart Coder - Smart Interviews (DSA)": {
        "skills": ["data structures", "algorithms", "competitive programming"],
        "patterns": [r"(?i)smart\s+coder\s+-?\s+smart\s+interviews\s+\(?dsa\)?"]
    },
    "Java (Basic) - HackerRank": {
        "skills": ["java", "object-oriented programming"],
        "patterns": [r"(?i)java\s*\(?basic\)?\s*-?\s*hacker\s*rank"]
    },
    "SQL (Basic) - HackerRank": {
        "skills": ["sql", "database management"],
        "patterns": [r"(?i)sql\s*\(?basic\)?\s*-?\s*hacker\s*rank"]
    },
    "AWS Certified Solutions Architect": {
        "skills": ["aws", "cloud computing", "networking"],
        "patterns": [r"(?i)aws\s+certified\s+solutions\s+architect", r"(?i)aws\s+certified\s+architect"]
    },
    "Cisco SQL Certification": {
        "skills": ["sql", "postgresql", "database design"],
        "patterns": [r"(?i)cisco\s+sql\s+certification"]
    },
    "GitHub Advanced Security Certified": {
        "skills": ["github security", "devops", "secure coding"],
        "patterns": [r"(?i)github\s+advanced\s+security\s+certified"]
    }
}


    def extract_text_from_resume(self, file_path):
        """Extract text from a PDF resume."""
        try:
            doc = fitz.open(file_path)
            return "\n".join([page.get_text("text") for page in doc])
        except Exception as e:
            print(f"Error extracting text: {e}")
            return None

    def extract_sections(self, text):
        """Extract sections like projects, skills, certifications."""
        sections = {}
        lines = text.split("\n")
        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            is_header = False
            for section_name, pattern in self.section_patterns.items():
                if re.match(pattern, line, re.IGNORECASE):
                    if current_section:
                        sections[current_section] = current_content
                    current_section = section_name
                    current_content = []
                    is_header = True
                    break

            if not is_header and current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = current_content

        return sections

    def extract_skills_and_projects(self, text):
      """Extracts skills and projects using regex-based matching."""
      sections = self.extract_sections(text)
      skills = set()
      projects = set()

      # Check for skills in the entire resume text
      for cert_info in self.cert_database.values():
          for skill in cert_info["skills"]:
              if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
                  skills.add(skill)

      # Check for projects in "Projects" section
      project_text = " ".join(sections.get('projects', []))
      for cert_info in self.cert_database.values():
          for project in cert_info.get("projects", []):  # ✅ Use .get() to avoid KeyError
              if re.search(rf"\b{re.escape(project)}\b", project_text, re.IGNORECASE):
                  projects.add(project)

      return {
          "skills": list(skills),
          "projects": list(projects)
      }



    def extract_certifications(self, text):
        """Extracts certifications using regex and similarity matching."""
        certifications = []

        for cert_name, cert_info in self.cert_database.items():
            # Check regex patterns
            for pattern in cert_info["patterns"]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    certifications.append({
                        "name": cert_name,
                        "text": matches,
                        "confidence": 0.9
                    })
                    break  # Avoid duplicate matches

            # Semantic matching using SBERT
            resume_embedding = self.sbert_model.encode(text, convert_to_tensor=True)
            cert_embedding = self.sbert_model.encode(cert_name, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(cert_embedding, resume_embedding).item()

            if similarity_score > 0.7:  # Accept if similarity is high
                certifications.append({
                    "name": cert_name,
                    "text": [cert_name],
                    "confidence": round(similarity_score, 2)
                })

        return certifications

    def validate_certification(self, certification, skills, projects):
      """Validates certification based on extracted skills and projects."""
      if certification["name"] not in self.cert_database:
          return 0.0

      cert_info = self.cert_database[certification["name"]]

      # ✅ Use .get() to avoid KeyError
      skill_score = len(set(skills) & set(cert_info.get("skills", []))) / max(1, len(cert_info.get("skills", [])))
      project_score = len(set(projects) & set(cert_info.get("projects", []))) / max(1, len(cert_info.get("projects", [])))

      return max(skill_score, project_score)


    def process_resume(self, file_path):
        """Extracts and validates skills, projects, and certifications."""
        text = self.extract_text_from_resume(file_path)
        if not text:
            return {"error": "Text extraction failed"}

        experience = self.extract_skills_and_projects(text)
        certifications = self.extract_certifications(text)

        results = []
        for cert in certifications:
            confidence = self.validate_certification(cert, experience["skills"], experience["projects"])
            results.append({
                "certification": cert["name"],
                "text": cert["text"],
                "is_valid":True if confidence >= 0.5 else False,
                "confidence": confidence
            })
        
        return {
    "certifications": results,
    "skills": experience["skills"],
    "projects": experience["projects"],
}


# # Example Usage
# if __name__ == "__main__":
#     validator = ResumeCertificationValidator()
#     results = validator.process_resume("/content/Palvai's Resume-hackerresume.pdf")
#     print(json.dumps(results, indent=2))