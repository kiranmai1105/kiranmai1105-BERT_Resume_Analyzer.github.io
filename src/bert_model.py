from sentence_transformers import SentenceTransformer, util

# Load pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Categories
CATEGORIES = {
    "Data Science": "machine learning data analysis python pandas numpy deep learning nlp",
    "Backend Development": "java spring boot flask django rest api backend server database",
    "Web Development": "html css javascript react frontend ui ux",
    "HR": "recruitment hiring onboarding employee management",
    "Database": "sql mysql database optimization queries"
}

category_names = list(CATEGORIES.keys())
category_desc = list(CATEGORIES.values())

category_embeddings = model.encode(category_desc, convert_to_tensor=True)


def predict_resume_bert(resume_text):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)

    scores = util.cos_sim(resume_embedding, category_embeddings)[0]

    best_idx = scores.argmax().item()
    confidence = scores[best_idx].item() * 100

    return category_names[best_idx], confidence
def predict_resume_bert_top3(resume_text):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)

    scores = util.cos_sim(resume_embedding, category_embeddings)[0]

    top_idx = scores.argsort(descending=True)[:3]

    results = []
    for i in top_idx:
        results.append((category_names[i], scores[i].item()*100))

    return results