# iris.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # to save model

# 1. Load data
df = pd.read_csv("data/Iris.csv")

# Drop ID column if present
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

print("Columns:", df.columns)
print("\nFirst 5 rows:\n", df.head())
print("\nClass distribution:\n", df["Species"].value_counts())

# 2. Visualization (optional)
sns.pairplot(df, hue="Species")
plt.show()

# Correlation heatmap
sns.heatmap(df.drop(columns="Species").corr(), annot=True, cmap="coolwarm")
plt.show()

# 3. Split data
X = df.drop("Species", axis=1)   # features
y = df["Species"]                # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train model (Logistic Regression as baseline)
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_scaled)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save trained model + scaler
joblib.dump(model, "models/iris_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\nModel and scaler saved in models/ folder.")


