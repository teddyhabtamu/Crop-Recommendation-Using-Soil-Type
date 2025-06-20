import os
import requests
import logging
import time
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
import pandas as pd
import pickle
import pymongo
from bson.objectid import ObjectId
from datetime import datetime, timezone
import bcrypt
from dotenv import load_dotenv
import google.generativeai as genai
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
esp32_url = os.getenv("ESP32_URL")  # e.g., http://192.168.1.100/read
app_secret = os.getenv("SECRET_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = app_secret

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# In-memory cache for Gemini reasons
reason_cache = {}

# Configure Gemini API
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logging.info("Gemini API configured successfully with gemini-1.5-flash")
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {str(e)}")
        gemini_model = None
else:
    logging.warning("GEMINI_API_KEY not found - using fallback reasons")
    gemini_model = None

# MongoDB connection
try:
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client["SensorData"]
    users_collection = db["Users"]
    readings_collection = db["Readings"]
    client.server_info()
    logging.info("Successfully connected to MongoDB")
except pymongo.errors.ServerSelectionTimeoutError as e:
    print(f"Connection timed out: {e}")
    exit(1)
except pymongo.errors.ConfigurationError as e:
    print(f"Configuration error: {e}")
    exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    exit(1)

# Load ML models
base_dir = os.path.dirname(__file__)
try:
    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    model = pickle.load(open(os.path.join(base_dir, 'model.pkl'), 'rb'))
    sc = pickle.load(open(os.path.join(base_dir, 'standscaler.pkl'), 'rb'))
    le = pickle.load(open(os.path.join(base_dir, 'labelencoder.pkl'), 'rb'))
    logging.info("ML models loaded successfully")
except FileNotFoundError as e:
    logging.error(f"Failed to load ML model files: {e}")
    exit(1)

# Crop dictionary (maps numerical labels to crop names)
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Recommendation function
def recommendation(features):
    try:
        logging.info(f"Input features: {features}")
        # Convert features to DataFrame with column names to match StandardScaler
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph']
        features_df = pd.DataFrame(features, columns=feature_names)
        sc_features = sc.transform(features_df)
        logging.info(f"Scaled features: {sc_features}")
        probabilities = model.predict_proba(sc_features)[0]
        logging.info(f"Probabilities: {probabilities}")
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        logging.info(f"Top 3 indices: {top_3_indices}")
        
        # Map indices to numerical labels using LabelEncoder
        top_3_numerical = le.inverse_transform(top_3_indices)
        logging.info(f"Top 3 numerical labels: {top_3_numerical}")
        
        # Map numerical labels to crop names using crop_dict
        top_3_crops = [crop_dict.get(int(label), "Unknown") for label in top_3_numerical]
        if "Unknown" in top_3_crops:
            logging.error(f"Invalid labels in top_3_numerical: {top_3_numerical}")
        
        logging.info(f"Top 3 crops: {top_3_crops}")
        return top_3_crops
    except Exception as e:
        logging.error(f"Error in recommendation: {str(e)}")
        raise

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if user:
        return User(str(user["_id"]), user["username"])
    return None

def generate_reason(crop, nitrogen, phosphorus, potassium, temperature, humidity, ph):
    cache_key = f"{crop}:{nitrogen}:{phosphorus}:{potassium}:{temperature}:{humidity}:{ph}"
    if cache_key in reason_cache:
        logging.info(f"Using cached reason for {cache_key}")
        return reason_cache[cache_key]

    try:
        if gemini_model:
            prompt = f"""
            As an agricultural expert, explain in 2-3 concise sentences why {crop} is ideal for cultivation with:
            - Nitrogen: {nitrogen} mg/kg
            - Phosphorus: {phosphorus} mg/kg
            - Potassium: {potassium} mg/kg
            - Temperature: {temperature}°C
            - Humidity: {humidity}%
            - Soil pH: {ph}
            Focus on how these specific conditions benefit {crop} growth. Use simple language suitable for farmers.
            """
            for attempt in range(3):
                try:
                    response = gemini_model.generate_content(
                        prompt,
                        generation_config={"temperature": 0.7, "max_output_tokens": 200}
                    )
                    reason = response.text.strip()
                    reason_cache[cache_key] = reason
                    logging.info(f"Generated reason for {crop}: {reason}")
                    return reason
                except Exception as e:
                    if "429" in str(e):
                        logging.warning(f"Gemini API 429 error, retrying in {2 ** attempt} seconds...")
                        time.sleep(2 ** attempt)
                    else:
                        raise e
            logging.error("Gemini API quota exceeded after retries")
    
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")

    fallback_reasons = {
        "Rice": f"Rice thrives with high nitrogen ({nitrogen} mg/kg), warm temperatures ({temperature}°C), and high humidity ({humidity}%). The soil pH of {ph} supports its waterlogged growth.",
        "Maize": f"Maize grows well with balanced nutrients (N:{nitrogen}, P:{phosphorus}, K:{potassium}) and warm temperatures ({temperature}°C). The pH of {ph} is ideal for its root development.",
        "Jute": f"Jute prefers high humidity ({humidity}%) and warm temperatures ({temperature}°C). The nutrient levels (N:{nitrogen}, P:{phosphorus}) suit its fiber production.",
        "Cotton": f"Cotton benefits from moderate nitrogen ({nitrogen} mg/kg) and potassium ({potassium} mg/kg) with a pH of {ph}. The temperature ({temperature}°C) supports its boll formation.",
        "Coconut": f"Coconut trees flourish in humid ({humidity}%) and warm ({temperature}°C) conditions. The soil pH of {ph} aids nutrient uptake.",
        "Papaya": f"Papaya grows well with high potassium ({potassium} mg/kg) and warm temperatures ({temperature}°C). The pH of {ph} ensures healthy fruit development.",
        "Orange": f"Orange trees thrive with moderate nutrients (N:{nitrogen}, P:{phosphorus}) and a pH of {ph}. The temperature ({temperature}°C) is ideal for fruit ripening.",
        "Apple": f"Apples prefer cooler temperatures ({temperature}°C) and balanced nutrients (N:{nitrogen}, K:{potassium}). The pH of {ph} supports tree health.",
        "Muskmelon": f"Muskmelon grows well with high potassium ({potassium} mg/kg) and warm temperatures ({temperature}°C). The pH of {ph} aids sweet fruit production.",
        "Watermelon": f"Watermelon thrives in warm ({temperature}°C) and moderately humid ({humidity}%) conditions. The nutrient levels (N:{nitrogen}, P:{phosphorus}) support large fruit growth.",
        "Grapes": f"Grapes prefer moderate nutrients (N:{nitrogen}, K:{potassium}) and a pH of {ph}. The temperature ({temperature}°C) is suitable for vine development.",
        "Mango": f"Mango trees flourish in warm ({temperature}°C) and humid ({humidity}%) conditions. The pH of {ph} supports fruit quality.",
        "Banana": f"Bananas grow well with high potassium ({potassium} mg/kg) and warm temperatures ({temperature}°C). The pH of {ph} aids robust growth.",
        "Pomegranate": f"Pomegranate thrives with moderate nutrients (N:{nitrogen}, P:{phosphorus}) and a pH of {ph}. The temperature ({temperature}°C) supports fruit ripening.",
        "Lentil": f"Lentils prefer cooler temperatures ({temperature}°C) and balanced nutrients (N:{nitrogen}, P:{phosphorus}). The pH of {ph} ensures healthy pod formation.",
        "Blackgram": f"Blackgram grows well with moderate nitrogen ({nitrogen} mg/kg) and a pH of {ph}. The temperature ({temperature}°C) supports its growth cycle.",
        "Mungbean": f"Mungbean thrives with low nitrogen ({nitrogen} mg/kg) and warm temperatures ({temperature}°C). The pH of {ph} aids seed development.",
        "Mothbeans": f"Mothbeans prefer high temperatures ({temperature}°C) and low nutrients (N:{nitrogen}). The pH of {ph} supports drought-tolerant growth.",
        "Pigeonpeas": f"Pigeonpeas grow well with moderate nutrients (N:{nitrogen}, P:{phosphorus}) and a pH of {ph}. The temperature ({temperature}°C) is ideal.",
        "Kidneybeans": f"Kidneybeans thrive with balanced nutrients (N:{nitrogen}, K:{potassium}) and a pH of {ph}. The temperature ({temperature}°C) supports pod growth.",
        "Chickpea": f"Chickpeas prefer cooler temperatures ({temperature}°C) and low nitrogen ({nitrogen} mg/kg). The pH of {ph} aids seed production.",
        "Coffee": f"Coffee plants flourish in humid ({humidity}%) and warm ({temperature}°C) conditions. The soil pH of {ph} supports berry development."
    }
    reason = fallback_reasons.get(crop, 
        f"{crop} is recommended because the conditions (N:{nitrogen}, P:{phosphorus}, K:{potassium}, Temp:{temperature}°C, Humidity:{humidity}%, pH:{ph}) are suitable for its growth.")
    reason_cache[cache_key] = reason
    return reason

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users_collection.find_one({"username": username}):
            flash("Username already exists", "error")
            return redirect(url_for('register'))
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        users_collection.insert_one({
            "username": username,
            "password_hash": password_hash
        })
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({"username": username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user["password_hash"].encode('utf-8')):
            user_obj = User(str(user["_id"]), user["username"])
            login_user(user_obj)
            flash("Login successful!", "success")
            return redirect(url_for('index'))
        flash("Invalid username or password", "error")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    page = int(request.args.get('page', 1))
    per_page = 7
    skip = (page - 1) * per_page
    total_readings = readings_collection.count_documents({"username": current_user.username})
    total_pages = (total_readings + per_page - 1) // per_page
    
    readings = list(readings_collection.find({"username": current_user.username})
                   .sort("timestamp", -1)
                   .skip(skip)
                   .limit(per_page))
    eat = pytz.timezone('Africa/Nairobi')  # Same as Addis Ababa timezone
    for reading in readings:
        if reading['timestamp'].tzinfo is None:
            # If timestamp is naive, assume it's UTC
            reading['timestamp'] = pytz.utc.localize(reading['timestamp'])
        reading['timestamp'] = reading['timestamp'].astimezone(eat)

    
    return render_template("index.html", 
                         readings=readings, 
                         username=current_user.username,
                         current_page=page,
                         total_pages=total_pages)

@app.route('/get_reason/<reading_id>', methods=['GET'])
@login_required
def get_reason(reading_id):
    try:
        reading = readings_collection.find_one({"_id": ObjectId(reading_id), "username": current_user.username})
        if not reading:
            logging.error(f"Reading {reading_id} not found or unauthorized")
            return jsonify({"error": "Reading not found or unauthorized"}), 404
        
        crops = reading.get("crops", ["Unknown"])
        reasons = reading.get("reasons", ["No reason available."] * len(crops))
        image_files = []
        for crop in crops:
            image_file = f"{crop.lower()}.png"
            static_dir = os.path.join(base_dir, 'static')
            if not os.path.exists(os.path.join(static_dir, image_file)):
                image_file = "default.png"
                logging.warning(f"Image not found for {crop}, using default: {image_file}")
            image_files.append(image_file)
        
        result = f"Top crops: {', '.join(crops)}" if crops != ["Unknown"] else "Could not determine the best crops."
        zipped_results = list(zip(crops, image_files, reasons))
        
        logging.info(f"Fetched reasons for reading {reading_id}: {reasons}")
        return jsonify({
            "zipped_results": [{"crop": crop, "image": image, "reason": reason} for crop, image, reason in zipped_results],
            "result": result
        })
    except Exception as e:
        logging.error(f"Error fetching reason for reading {reading_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/request_reading', methods=['POST'])
@login_required
def request_reading():
    try:
        # Request data from ESP32
        esp32_response = requests.post(
            esp32_url,
            json={"username": current_user.username},
            timeout=10
        )
        if esp32_response.status_code != 200:
            logging.error(f"ESP32 request failed with status {esp32_response.status_code}")
            flash(f"Failed to get sensor data: {esp32_response.status_code}", "error")
            return redirect(url_for('index', page=request.args.get('page', 1)))

        data = esp32_response.json()
        required_fields = ['username', 'temperature', 'humidity', 'ph', 'nitrogen', 'phosphorus', 'potassium']
        if not all(field in data for field in required_fields):
            logging.error(f"Invalid sensor data: {data}")
            flash("Invalid sensor data", "error")
            return redirect(url_for('index', page=request.args.get('page', 1)))
        if data['username'] != current_user.username:
            logging.error(f"Username mismatch: expected {current_user.username}, got {data['username']}")
            flash("Username mismatch", "error")
            return redirect(url_for('index', page=request.args.get('page', 1)))

        timestamp = datetime.now(timezone.utc)
        reading = {
            "username": data['username'],
            "temperature": float(data['temperature']),
            "humidity": float(data['humidity']),
            "ph": float(data['ph']),
            "nitrogen": float(data['nitrogen']),
            "phosphorus": float(data['phosphorus']),
            "potassium": float(data['potassium']),
            "timestamp": timestamp
        }

        # Generate prediction
        feature_list = [
            reading['nitrogen'], 
            reading['phosphorus'], 
            reading['potassium'],
            reading['temperature'], 
            reading['humidity'], 
            reading['ph']
        ]
        single_pred = np.array(feature_list).reshape(1, -1)
        crops = recommendation(single_pred)

        # Prepare images and reasons
        image_files = []
        for crop in crops:
            image_file = f"{crop.lower()}.png"
            static_dir = os.path.join(base_dir, 'static')
            if not os.path.exists(os.path.join(static_dir, image_file)):
                image_file = "default.png"
                logging.warning(f"Image not found for {crop}, using default: {image_file}")
            image_files.append(image_file)

        reasons = [
            generate_reason(
                crop, 
                reading['nitrogen'], 
                reading['phosphorus'], 
                reading['potassium'],
                reading['temperature'], 
                reading['humidity'], 
                reading['ph']
            ) for crop in crops
        ]
        result = f"Top crops: {', '.join(crops)}" if crops != ["Unknown"] else "Could not determine the best crops."
        zipped_results = list(zip(crops, image_files, reasons))

        # Store reading with prediction
        reading['crops'] = crops
        reading['reasons'] = reasons
        readings_collection.insert_one(reading)

        # Fetch updated readings for the table
        page = int(request.args.get('page', 1))
        per_page = 7
        skip = (page - 1) * per_page
        total_readings = readings_collection.count_documents({"username": current_user.username})
        total_pages = (total_readings + per_page - 1) // per_page
        readings = list(readings_collection.find({"username": current_user.username})
                       .sort("timestamp", -1)
                       .skip(skip)
                       .limit(per_page))

        # Localize timestamps to EAT
        eat = pytz.timezone('Africa/Nairobi')
        for reading in readings:
            if reading['timestamp'].tzinfo is None:
                reading['timestamp'] = pytz.utc.localize(reading['timestamp'])
            reading['timestamp'] = reading['timestamp'].astimezone(eat)

        # Render template with prediction results
        return render_template(
            'index.html',
            result=result,
            zipped_results=zipped_results,
            readings=readings,
            username=current_user.username,
            current_page=page,
            total_pages=total_pages
        )

    except requests.RequestException as e:
        logging.error(f"ESP32 connection error: {str(e)}")
        flash(f"Failed to connect to ESP32: {str(e)}", "error")
        return redirect(url_for('index', page=request.args.get('page', 1)))
    except Exception as e:
        logging.error(f"Server error in request_reading: {str(e)}")
        flash(f"Server error: {str(e)}", "error")
        return redirect(url_for('index', page=request.args.get('page', 1)))

@app.route("/predict", methods=['POST'])
@login_required
def predict():
    try:
        # Safely retrieve form data with defaults to avoid KeyError
        N = float(request.form.get('Nitrogen', '0'))
        P = float(request.form.get('Phosphorus', '0'))  # Corrected typo from 'Phosporus' to 'Phosphorus'
        K = float(request.form.get('Potassium', '0'))
        temp = float(request.form.get('Temperature', '0'))
        humidity = float(request.form.get('Humidity', '0'))
        ph = float(request.form.get('pH', '0'))
        page = int(request.form.get('page', '1'))

        # Validate input ranges (optional, can be moved to client-side if preferred)
        if not (0 <= N <= 140 and 0 <= P <= 120 and 0 <= K <= 200 and
                10 <= temp <= 45 and 0 <= humidity <= 100 and 3.5 <= ph <= 9.0):
            raise ValueError("Input values out of valid range")

        feature_list = [N, P, K, temp, humidity, ph]
        single_pred = np.array(feature_list).reshape(1, -1)
        crops = recommendation(single_pred)

        image_files = []
        for crop in crops:
            image_file = f"{crop.lower()}.png"
            static_dir = os.path.join(base_dir, 'static')
            if not os.path.exists(os.path.join(static_dir, image_file)):
                image_file = "default.png"
                logging.warning(f"Image not found for {crop}, using default: {image_file}")
            image_files.append(image_file)

        reasons = [generate_reason(crop, N, P, K, temp, humidity, ph) for crop in crops]
        result = f"Top crops: {', '.join(crops)}" if crops != ["Unknown"] else "Could not determine the best crops."
        zipped_results = list(zip(crops, image_files, reasons))

        timestamp = datetime.now(pytz.utc)
        logging.info(f"Storing manual prediction with timestamp: {timestamp.strftime('%Y-%m-%d %I:%M %p %Z')}")
        readings_collection.insert_one({
            "username": current_user.username,
            "nitrogen": N,
            "phosphorus": P,
            "potassium": K,
            "temperature": temp,
            "humidity": humidity,
            "ph": ph,
            "crops": crops,
            "reasons": reasons,
            "timestamp": timestamp
        })

        per_page = 7
        skip = (page - 1) * per_page
        total_readings = readings_collection.count_documents({"username": current_user.username})
        total_pages = (total_readings + per_page - 1) // per_page
        readings = list(readings_collection.find({"username": current_user.username})
                       .sort("timestamp", -1)
                       .skip(skip)
                       .limit(per_page))
        
        eat = pytz.timezone('Africa/Nairobi')
        for reading in readings:
            if reading['timestamp'].tzinfo is None:
                reading['timestamp'] = pytz.utc.localize(reading['timestamp'])
            reading['timestamp'] = reading['timestamp'].astimezone(eat)

        return render_template(
            'index.html',
            result=result,
            zipped_results=zipped_results,
            readings=readings,
            username=current_user.username,
            current_page=page,
            total_pages=total_pages
        )

    except ValueError as e:
        logging.error(f"Input validation error: {str(e)}")
        flash(f"Invalid input: Please enter valid numbers within the allowed ranges.", "error")
        return redirect(url_for('index', page=request.form.get('page', 1)))
    except KeyError as e:
        logging.error(f"Missing form field: {str(e)}")
        flash(f"Error: Missing required input field. Please try again.", "error")
        return redirect(url_for('index', page=request.form.get('page', 1)))
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        flash(f"Prediction failed: Unable to process input. Please try again.", "error")
        return redirect(url_for('index', page=request.form.get('page', 1)))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
