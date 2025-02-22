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
            "Infosys Springboard Python certification": {
                "skills": ["python", "django", "flask", "numpy", "pandas"],
                "projects": ["python development", "ai models", "data science"],
                "patterns": [r"infosys\s+springboard\s+python\s+certification"]
            },
            "Infosys Springboard Java certification": {
                "skills": ["java", "spring boot", "hibernate"],
                "projects": ["enterprise applications", "backend development"],
                "patterns": [r"infosys\s+springboard\s+java\s+certification"]
            },
            "Cisco SQL certification": {
                "skills": ["sql", "mysql", "postgresql", "database design"],
                "projects": ["database management", "data warehousing"],
                "patterns": [r"cisco\s+sql\s+certification"]
            },
            "HackerRank Python (basic) certification": {
                "skills": ["python", "programming basics"],
                "projects": ["basic python development"],
                "patterns": [r"hacker\s*rank\s*python\s*\(?basic\)?\s*certification"]
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
            for project in cert_info["projects"]:
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

        # Calculate scores
        skill_score = len(set(skills) & set(cert_info["skills"])) / len(cert_info["skills"])
        project_score = len(set(projects) & set(cert_info["projects"])) / len(cert_info["projects"])

        return max(skill_score, project_score)

    def process_resume(self, file_path):
        """Extracts and validates skills, projects, and certifications."""
        text = self.extract_text_from_resume(file_path)
        if not text:
            return {"error": "Text extraction failed"}
    
        experience = self.extract_skills_and_projects(text)
        certifications = self.extract_certifications(text)
    
        results = []
        all_valid = True  # Flag to check if all certifications are valid
    
        for cert in certifications:
            confidence = self.validate_certification(cert, experience["skills"], experience["projects"])
            is_valid = confidence >= 0.5
            results.append({
                "certification": cert["name"],
                "text": cert["text"],
                "is_valid": is_valid,
                "confidence": confidence
            })
    
            if not is_valid:
                all_valid = False  # If any certification is invalid, set flag to False
    
        return {
            "all_valid": all_valid,
            "certifications": results,
            "skills": experience["skills"],
            "projects": experience["projects"]
        }
# Example Usage
if __name__ == "__main__":
    validator = ResumeCertificationValidator()
    results = validator.process_resume("Palvai's Resume-hackerresume.pdf")
    print(json.dumps(results, indent=3))