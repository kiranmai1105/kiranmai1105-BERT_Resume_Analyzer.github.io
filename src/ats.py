from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

SKILLS_DB = [
    "python", "java", "machine learning", "deep learning", "nlp",
    "sql", "flask", "django", "react", "javascript",
    "html", "css", "tensorflow", "pandas", "numpy",
    "spring boot", "rest api", "data analysis", "ai",
    "model deployment", "data structures", "algorithms"
]


def extract_skills(text):
    text = text.lower()
    return list(set([skill for skill in SKILLS_DB if skill in text]))


def compute_ats_score(resume_text, jd_text):
    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(jd_text, convert_to_tensor=True)

    similarity = util.cos_sim(emb1, emb2)[0][0].item()
    score = similarity * 100

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    matched = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))

    return score, matched, missing