# 📌 Email Spam Detector using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# 2. Keep only the useful columns (drop extra columns if any)
df = df[['v1', 'v2']].copy()
df.columns = ['label', 'message']

# 3. Drop missing values (if any)
df.dropna(inplace=True)

# 4. Encode labels: ham -> 0, spam -> 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 5. Split data into features and labels
X = df['message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test_vec)

print("📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# 9. Test with custom messages
def predict_spam(msg):
    msg_vec = vectorizer.transform([msg])
    pred = model.predict(msg_vec)[0]
    return "🚨 Spam" if pred == 1 else "✅ Not Spam"

print("\n🔎 Custom Tests:")
test_messages = [
    "Congratulations! You won a lottery of $10,000. Claim now!",
    "Hi John, are we meeting at 5 pm tomorrow?",
    "URGENT: Your account has been compromised. Click here to reset your password!",
    "Don't miss our limited time offer. Buy one get one free!",
    "Hey, can you send me the notes for class today?",
    "You have been selected for a free gift card worth $500.",
    "Reminder: Your appointment with Dr. Smith is tomorrow at 10 am.",
    "Winner! Claim your cash prize before it expires.",
    "Are we still going for dinner tonight?",
    "Act now! Low interest loans available for a short time only."
]

for msg in test_messages:
    print(f"Message: {msg}\nPrediction: {predict_spam(msg)}\n")
