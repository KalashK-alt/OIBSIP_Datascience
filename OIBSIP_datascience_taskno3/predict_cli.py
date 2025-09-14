# src/predict_cli.py
"""
Small CLI to load the saved pipeline and predict price for a single car.
Example usage:
  python src/predict_cli.py --present_price 7.5 --driven_kms 20000 --owner 0 --year 2016 --car_name "ciaz" --fuel_type Petrol --selling_type Dealer --transmission Manual
"""
import argparse
import joblib
import pandas as pd
import os

PIPE_PATH = os.path.join("artifacts", "car_price_pipeline.pkl")
CURRENT_YEAR = 2025

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--present_price", type=float, required=True, help="Present (company) price")
    p.add_argument("--driven_kms", type=int, required=True)
    p.add_argument("--owner", type=int, default=0)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--car_name", type=str, required=True)
    p.add_argument("--fuel_type", type=str, choices=["Petrol", "Diesel", "CNG"], required=True)
    p.add_argument("--selling_type", type=str, choices=["Dealer", "Individual"], required=True)
    p.add_argument("--transmission", type=str, choices=["Manual", "Automatic"], required=True)
    return p.parse_args()

def main():
    args = parse_args()
    pipeline = joblib.load(PIPE_PATH)
    age = CURRENT_YEAR - args.year
    brand = args.car_name.split()[0].lower()
    row = {
        "Present_Price": args.present_price,
        "Driven_kms": args.driven_kms,
        "Owner": args.owner,
        "Age": age,
        "Fuel_Type": args.fuel_type,
        "Selling_type": args.selling_type,
        "Transmission": args.transmission,
        "Brand": brand
    }
    X = pd.DataFrame([row])
    pred = pipeline.predict(X)[0]
    print(f"Predicted selling price: {pred:.3f}")

if __name__ == "__main__":
    main()
