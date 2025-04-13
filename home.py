from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Simple home page
@app.route("/")
def home():
    return """
    <h2>House Price Prediction API is running!</h2>
    <p>Send a POST request to <code>/predict</code></p>
    <br>
    <form method="post" action="/form">
        <label>Area: <input type="number" step="0.1" name="area" required></label><br><br>
        <label>Bedrooms: <input type="number" name="bedrooms" required></label><br><br>
        <label>Bathrooms: <input type="number" name="bathrooms" required></label><br><br>
        <label>Year Built: <input type="number" name="yearbuilt" required></label><br><br>
        <label>Location: <input type="text" name="location" required></label><br><br>
        <input type="submit" value="Predict Price">
    </form>
    """

# POST request to API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        input_df = pd.DataFrame([{
            "area": data["area"],
            "bedrooms": data["bedrooms"],
            "bathrooms": data["bathrooms"],
            "yearbuilt": data["yearbuilt"],
            "location": data["location"]
        }])
        prediction = model.predict(input_df)[0]
        return jsonify({"predicted_price": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

# Handle form submission
@app.route("/form", methods=["POST"])
def form_predict():
    try:
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        yearbuilt = int(request.form["yearbuilt"])
        location = request.form["location"]

        input_df = pd.DataFrame([{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "yearbuilt": yearbuilt,
            "location": location
        }])

        predicted_price = model.predict(input_df)[0]
    except Exception as e:
        predicted_price = f"Error: {str(e)}"

    return render_template_string("""
    <h2>💰 Predicted Price: {{ price }}</h2>
    <a href="/">Go Back</a>
    """, price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
