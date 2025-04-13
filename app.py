from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "<h2>House Price Prediction API is running!</h2><p>Send a POST request to /predict</p>"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        # Construct DataFrame with correct column names
        df = pd.DataFrame([{
            "area": data["area"],
            "bedrooms": data["bedrooms"],
            "bathrooms": data["bathrooms"],
            "yearbuilt": data["yearbuilt"],
            "location": data["location"]
        }])
        
        prediction = model.predict(df)[0]
        return jsonify({"predicted_price": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
