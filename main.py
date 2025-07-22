import joblib
import numpy as np

# Load the saved model
model = joblib.load("prediction-model4-86.pkl")  

# Clean total_sqft string
def convert_sqft_to_num(x):
    try:
        x = str(x)
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

# Clean location string
def clean_location(location, known_locations):
    location = location.strip()
    return location if location in known_locations else "other"

# Example known locations (you must extract this from training set)
known_locations = [
    "Whitefield", "Sarjapur Road", "Electronic City", "HSR Layout", "Marathahalli",
    "other"  # include 'other'
]

# ---- INPUTS ----
print("🏠 Enter property details to get estimated price (in lakhs):")
location = input("Location: ")
total_sqft = input("Total area in sqft (e.g., 1000 or 900-1100): ")
bath = int(input("Number of bathrooms: "))
bhk = int(input("Number of bedrooms (BHK): "))

# ---- CLEANING ----
total_sqft = convert_sqft_to_num(total_sqft)
location = clean_location(location, known_locations)

if total_sqft is None:
    print("❌ Invalid sqft input. Please retry.")
    exit()

# ---- PREDICT ----
features = np.array([[location, total_sqft, bath, bhk]])
features1 = np.array([[12,2200,3,3]])
predicted_price = model.predict(features1)[0]

print(f"\n✅ Predicted Price: ₹ {predicted_price:.2f} Lakhs")
