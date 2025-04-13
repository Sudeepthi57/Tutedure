import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "area": 1600,
    "bedrooms": 3,
    "bathrooms": 2,
    "yearbuilt": 2012,
    "location": "Delhi"
}

print("Sending request...")
response = requests.post(url, json=data)
print(response.status_code)
print(response.json())
