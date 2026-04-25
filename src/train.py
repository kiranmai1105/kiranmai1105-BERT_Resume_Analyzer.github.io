import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from preprocess import clean_text
import pandas as pd
from bert_model import train_bert_classifier

df = pd.read_csv('data/resumes.csv')

texts = df['resume']
labels = df['category']

train_bert_classifier(texts, labels)

print("BERT model trained successfully!")

# Load data
df = pd.read_csv('data/resumes.csv')

# Preprocess
df['cleaned'] = df['resume'].apply(clean_text)

X = df['cleaned']
y = df['category']

# Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),   # VERY IMPORTANT
    stop_words='english'
)
X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Model

model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')