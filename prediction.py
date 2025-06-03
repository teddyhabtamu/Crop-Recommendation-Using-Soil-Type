import pickle
import numpy as np

# Load models
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
le = pickle.load(open('labelencoder.pkl', 'rb'))

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

print(le.classes_)

# Recommendation function
def recommendation(features):
    sc_features = sc.transform(features)
    probabilities = model.predict_proba(sc_features)[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_numerical = le.inverse_transform(top_3_indices)
    inv_crop_dict = {v: k for k, v in crop_dict.items()}
    top_3_crops = [inv_crop_dict[num] for num in top_3_numerical]
    return top_3_crops

# Test with your example inputs
features = np.array([[90, 42, 43, 20.879744, 82.002744, 6.502985]]).reshape(1, -1)
try:
    crops = recommendation(features)
    print("Top 3 crops:", crops)
except Exception as e:
    print("Error:", str(e))