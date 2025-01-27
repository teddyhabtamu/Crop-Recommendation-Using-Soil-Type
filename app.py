import os
from flask import Flask, request, render_template
import numpy as np
import pandas
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
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']

    feature_list = [N, P, K, temp, humidity, ph]
    single_pred = np.array(feature_list).reshape(1, -1)

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
        15: "lentil.png", 16: "blackgram.png", 17: "mungbean.png", 18: "mothbeans.png",
        19: "pigeonpeas.png", 20: "kidneybeans.png", 21: "chickpea.png", 22: "coffee.png"
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        image_file = crop_images.get(prediction[0], "default.png")  # Fallback image
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        crop = None
        image_file = "default.png"
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # Pass the crop name, result, and image to the template
    return render_template('index.html', result=result, crop=crop, image_file=image_file)


if __name__ == "__main__":
    app.run(debug=True)