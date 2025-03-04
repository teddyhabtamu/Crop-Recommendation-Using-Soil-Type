import os
from flask import Flask, request, render_template
import numpy as np
import pickle

# Define the base directory
base_dir = os.path.dirname(__file__)

# Construct full paths
model_path = os.path.join(base_dir, 'model.pkl')
sc_path = os.path.join(base_dir, 'standscaler.pkl')
mx_path = os.path.join(base_dir, 'minmaxscaler.pkl')

# Load the models
model = pickle.load(open(model_path, 'rb'))
sc = pickle.load(open(sc_path, 'rb'))
mx = pickle.load(open(mx_path, 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get form data
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])

    # Prepare feature list for prediction
    feature_list = [N, P, K, temp, humidity, ph]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Transform features using scalers
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    # Crop dictionary with names
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Dictionary for crop images
    crop_images = {
        1: "rice.png", 2: "maize.png", 3: "jute.png", 4: "cotton.png", 5: "coconut.png",
        6: "papaya.png", 7: "orange.png", 8: "apple.png", 9: "muskmelon.png", 10: "watermelon.png",
        11: "grapes.png", 12: "mango.png", 13: "banana.png", 14: "pomegranate.png",
        15: "lentil.png", 16: "blackgram.png", 17: "mungbeans.png", 18: "mothbeans.png",
        19: "pigeonpeas.png", 20: "kidneybeans.png", 21: "chickpea.png", 22: "coffee.png"
    }

    # Generate reason for recommendation
    def generate_reason(crop, N, P, K, temp, humidity, ph):
        reasons = {
            "Rice": "Rice is recommended because it thrives in high nitrogen levels, warm temperatures, and high humidity.",
            "Maize": "Maize is recommended due to its adaptability to moderate nitrogen, phosphorus, and potassium levels, along with warm temperatures.",
            "Jute": "Jute is suitable for areas with moderate nitrogen and potassium levels, and it prefers warm and humid conditions.",
            "Cotton": "Cotton is recommended for its ability to grow in well-drained soils with moderate nitrogen and phosphorus levels.",
            "Coconut": "Coconut is ideal for tropical climates with high humidity and moderate nutrient levels.",
            "Papaya": "Papaya thrives in warm temperatures, high humidity, and well-drained soils with balanced nutrients.",
            "Orange": "Oranges grow well in moderate temperatures, balanced pH, and adequate nitrogen and potassium levels.",
            "Apple": "Apples are recommended for cooler climates with moderate nitrogen and phosphorus levels.",
            "Muskmelon": "Muskmelons prefer warm temperatures, moderate humidity, and well-drained soils.",
            "Watermelon": "Watermelons thrive in warm climates with high humidity and balanced nutrient levels.",
            "Grapes": "Grapes are suitable for moderate temperatures, well-drained soils, and balanced nutrients.",
            "Mango": "Mangoes grow well in warm climates with moderate nitrogen and potassium levels.",
            "Banana": "Bananas are recommended for tropical climates with high humidity and balanced nutrients.",
            "Pomegranate": "Pomegranates thrive in warm temperatures, moderate humidity, and well-drained soils.",
            "Lentil": "Lentils are suitable for cooler climates with moderate nitrogen and phosphorus levels.",
            "Blackgram": "Blackgram grows well in warm temperatures and moderate nutrient levels.",
            "Mungbean": "Mungbeans thrive in warm climates with moderate nitrogen and potassium levels.",
            "Mothbeans": "Mothbeans are recommended for arid climates with low to moderate nutrient levels.",
            "Pigeonpeas": "Pigeonpeas grow well in warm temperatures and moderate nutrient levels.",
            "Kidneybeans": "Kidneybeans are suitable for moderate temperatures and balanced nutrients.",
            "Chickpea": "Chickpeas thrive in cooler climates with moderate nitrogen and phosphorus levels.",
            "Coffee": "Coffee is recommended for tropical climates with high humidity and balanced nutrients."
        }
        return reasons.get(crop, "This crop is recommended based on the provided soil and environmental conditions.")

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        image_file = crop_images.get(prediction[0], "default.png")  # Fallback image
        result = "{} is the best crop to be cultivated right there".format(crop)
        reason = generate_reason(crop, N, P, K, temp, humidity, ph)
    else:
        crop = None
        image_file = "default.png"
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        reason = "No specific reason available due to insufficient data."

    # Pass the crop name, result, image, and reason to the template
    return render_template('index.html', result=result, crop=crop, image_file=image_file, reason=reason)

if __name__ == "__main__":
    app.run(debug=True)