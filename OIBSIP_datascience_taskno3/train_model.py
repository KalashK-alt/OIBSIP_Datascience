# src/train_model.py
"""
Train a Car Price prediction pipeline and save it to artifacts/car_price_pipeline.pkl

Usage:
    python src/train_model.py
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = os.path.join("data", "car data.csv")
SAVE_DIR = os.path.join("artifacts")
SAVE_PATH = os.path.join(SAVE_DIR, "car_price_pipeline.pkl")
CURRENT_YEAR = 2025 

def load_and_prepare(path):
    df = pd.read_csv(path)
    # Feature engineering
    df["Age"] = CURRENT_YEAR - df["Year"]
    # extract brand (simple heuristic: first token)
    df["Brand"] = df["Car_Name"].apply(lambda x: str(x).split()[0].lower())
    # select features
    features = ["Present_Price", "Driven_kms", "Owner", "Age",
                "Fuel_Type", "Selling_type", "Transmission", "Brand"]
    X = df[features].copy()
    y = df["Selling_Price"].copy()
    return X, y

def build_pipeline():
    num_features = ["Present_Price", "Driven_kms", "Owner", "Age"]
    cat_features = ["Fuel_Type", "Selling_type", "Transmission", "Brand"]

    num_transformer = Pipeline([("scaler", StandardScaler())])
    cat_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ])

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline, num_features, cat_features

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    X, y = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    pipeline, num_feats, cat_feats = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

    # Save pipeline
    joblib.dump(pipeline, SAVE_PATH)

    # Extract feature names and importances
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    ohe_feature_names = list(ohe.get_feature_names_out(cat_feats))
    feature_names = num_feats + ohe_feature_names
    importances = pipeline.named_steps["model"].feature_importances_

    # Print summary
    print("Saved pipeline to:", SAVE_PATH)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")
    print("5-fold CV R2 scores:", cv_scores)
    print("CV R2 mean: {:.4f}".format(cv_scores.mean()))

    # show top importances
    import pandas as pd
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop 15 feature importances:\n", fi.head(15))

if __name__ == "__main__":
    main()
