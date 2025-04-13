import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load data
df = pd.read_csv("house_data.csv")
print("Data loaded successfully.")

# Rename columns for consistency
df.rename(columns={
    "Area (sqft)": "area",
    "Bedrooms": "bedrooms",
    "Bathrooms": "bathrooms",
    "Location": "location",
    "YearBuilt": "yearbuilt",
    "Price": "price"
}, inplace=True)

# Print columns after renaming
print("Columns renamed:", df.columns)

# Define features and target
X = df[["area", "bedrooms", "bathrooms", "yearbuilt", "location"]]
y = df["price"]

# Check the shape of the data
print(f"Data shape: {X.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Create a column transformer with OneHotEncoder for categorical features
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["location"])],
    remainder="passthrough"  # Keep the numerical features as is
)

# Create a pipeline that first transforms the data then fits the model
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Fit the model
model_pipeline.fit(X_train, y_train)
print("Model training completed.")

# Save the model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("Model saved to 'model.pkl'.")