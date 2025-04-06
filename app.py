# import os
# print("Current working directory:", os.getcwd())
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# from model.drug_model import DrugProteinModel 

# app = Flask(__name__)
# CORS(app)

# model = joblib.load("model/sentiment_model.pkl")

# from utils.predict import get_affinity

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     drug = data.get("drug")
#     target = data.get("target")

#     if not drug or not target:
#         return jsonify({"error": "Missing drug or target"}), 400

#     affinity_score = get_affinity(model, drug, target)
#     return jsonify({"affinity": float(affinity_score)})

    
# @app.route("/", methods=["GET"])
# def home():
#     return "API is running successfully! 🚀"

# if __name__ == "__main__":
#     app.run(debug=True)

