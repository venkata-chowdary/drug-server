
import os
import torch
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from model.drug_model import DrugProteinModel  # :contentReference[oaicite:0]{index=0}
from utils.predict import get_affinity           # :contentReference[oaicite:1]{index=1}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the CSV file with actual KIBA scores
csv_path = "KIBA.csv"
if os.path.exists(csv_path):
    kiba_df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(kiba_df)} rows.")
else:
    kiba_df = None
    print("Warning: KIBA.csv file not found. Actual KIBA scores won't be available.")

# Define the path to your saved model
# model_path = "model/model.pkl"
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file not found at {model_path}")


drug_in_channels = 5  # number of features per atom in SMILES graph (you had 5 features: atomic num, degree, etc.)
protein_embedding_size = 128  # your random protein embedding size
hidden_channels = 64  # example value, check what you used
out_channels = 1  # if you predict a single KIBA score (regression)

# Define the path to your saved model
model_path = "model/model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the trained model using joblib
model = joblib.load(model_path)
model.eval()  # Set the model to evaluation mode


# Create model instance with correct parameters
# model = DrugProteinModel(
#     drug_in_channels=drug_in_channels,
#     protein_embedding_size=protein_embedding_size,
#     hidden_channels=hidden_channels,
#     out_channels=out_channels
# )


@app.route("/", methods=["GET"])
def index():
    return "Drug-Target Affinity API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    drug = data.get("drug")
    target = data.get("target")
    
    if not drug or not target:
        return jsonify({"error": "Both 'drug' and 'target' fields are required."}), 400
    
    try:

        # Look up the actual KIBA value from the CSV file
        actual_kiba = None
        if kiba_df is not None:
            # Ensure the column names exactly match your CSV
            row = kiba_df[(kiba_df['compound_iso_smiles'] == drug) & (kiba_df['target_sequence'] == target)]
            if not row.empty:
                actual_kiba = row.iloc[0]['Ki , Kd and IC50  (KIBA Score)']
        
        if actual_kiba is None:
            actual_str = "not available"
        else:
            try:
                actual_kiba = float(actual_kiba)
                actual_str = f"{actual_kiba:.4f}"
            except Exception:
                actual_str = str(actual_kiba)
            
            # Get the predicted KIBA value from your model
            predicted = get_affinity(model, drug, target,actual_str)
    
            response_text = f"Predicted KIBA: {predicted:.4f}, Actual KIBA: {actual_str}"
            print(response_text)
            return jsonify({"predicted": f"{predicted:.4f}","actual": actual_str})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
