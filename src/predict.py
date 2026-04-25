import joblib
from preprocess import clean_text

model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

def predict_resume(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    # Sort top predictions
    top_indices = probs.argsort()[-3:][::-1]

    top_predictions = [(classes[i], probs[i]*100) for i in top_indices]

    return top_predictions