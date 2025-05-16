import os
import requests
import logging
import time
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
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
except pymongo.errors.ConnectionError as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
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
    mx = pickle.load(open(os.path.join(base_dir, 'minmaxscaler.pkl'), 'rb'))
    logging.info("ML models loaded successfully")
except FileNotFoundError as e:
    logging.error(f"Failed to load ML model files: {e}")
    exit(1)

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
    
    # Get readings and convert timestamps to EAT
    readings = list(readings_collection.find({"username": current_user.username})
               .sort("timestamp", -1)
               .skip(skip)
               .limit(per_page))
    
    # Convert UTC timestamps to EAT
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
        
        crop = reading.get("crop", "Unknown")
        reason = reading.get("reason", "No reason available.")
        image_file = f"{crop.lower()}.png"
        static_dir = os.path.join(base_dir, 'static')
        if not os.path.exists(os.path.join(static_dir, image_file)):
            image_file = "default.png"
            logging.warning(f"Image not found, using default: {image_file}")
        result = f"{crop} is the best crop to be cultivated." if crop != "Unknown" else "Could not determine the best crop."
        
        logging.info(f"Fetched reason for reading {reading_id}: {reason}")
        return jsonify({
            "crop": crop,
            "image_file": image_file,
            "result": result,
            "reason": reason
        })
    except Exception as e:
        logging.error(f"Error fetching reason for reading {reading_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/request_reading', methods=['POST'])
@login_required
def request_reading():
    try:
        # Send username to ESP32
        esp32_response = requests.post(
            esp32_url,
            json={"username": current_user.username},
            timeout=10
        )
        if esp32_response.status_code != 200:
            logging.error(f"ESP32 request failed with status {esp32_response.status_code}")
            return jsonify({"error": f"Failed to get sensor data: {esp32_response.status_code}"}), 500

        data = esp32_response.json()
        required_fields = ['username', 'temperature', 'humidity', 'ph', 'nitrogen', 'phosphorus', 'potassium']
        if not all(field in data for field in required_fields):
            logging.error(f"Invalid sensor data: {data}")
            return jsonify({"error": "Invalid sensor data"}), 400
        if data['username'] != current_user.username:
            logging.error(f"Username mismatch: expected {current_user.username}, got {data['username']}")
            return jsonify({"error": "Username mismatch"}), 401

        # Prepare reading
        timestamp = datetime.now(timezone.utc)
        logging.info(f"Storing reading with timestamp: {timestamp.strftime('%Y-%m-%d %I:%M %p %Z')}")
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

        # Run ML prediction
        feature_list = [
            reading['nitrogen'], 
            reading['phosphorus'], 
            reading['potassium'],
            reading['temperature'], 
            reading['humidity'], 
            reading['ph']
        ]
        single_pred = np.array(feature_list).reshape(1, -1)
        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }
        crop = crop_dict.get(prediction[0], "Unknown")
        image_file = f"{crop.lower()}.png"
        
        static_dir = os.path.join(base_dir, 'static')
        if not os.path.exists(os.path.join(static_dir, image_file)):
            image_file = "default.png"
            logging.warning(f"Image not found, using default: {image_file}")

        reason = generate_reason(
            crop, 
            reading['nitrogen'], 
            reading['phosphorus'], 
            reading['potassium'],
            reading['temperature'], 
            reading['humidity'], 
            reading['ph']
        )
        result = f"{crop} is the best crop to be cultivated."

        # Save reading with predicted crop and reason
        reading['crop'] = crop
        reading['reason'] = reason
        readings_collection.insert_one(reading)

        # Fetch updated readings for the current page
        page = int(request.args.get('page', 1))
        per_page = 7
        skip = (page - 1) * per_page
        total_readings = readings_collection.count_documents({"username": current_user.username})
        total_pages = (total_readings + per_page - 1) // per_page
        readings = list(readings_collection.find({"username": current_user.username})
                       .sort("timestamp", -1)
                       .skip(skip)
                       .limit(per_page))

        # Render index.html with prediction and updated readings
        return render_template(
            'index.html',
            result=result,
            crop=crop,
            image_file=image_file,
            reason=reason,
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
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        page = int(request.form.get('page', 1))

        feature_list = [N, P, K, temp, humidity, ph]
        single_pred = np.array(feature_list).reshape(1, -1)
        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }
        crop = crop_dict.get(prediction[0], None)
        image_file = f"{crop.lower()}.png" if crop else "default.png"
        
        static_dir = os.path.join(base_dir, 'static')
        if not os.path.exists(os.path.join(static_dir, image_file)):
            image_file = "default.png"
            logging.warning(f"Image not found, using default: {image_file}")

        if crop:
            result = f"{crop} is the best crop to be cultivated."
            reason = generate_reason(crop, N, P, K, temp, humidity, ph)
        else:
            result = "Could not determine the best crop."
            reason = "Insufficient data for recommendation."

        timestamp = datetime.now(timezone.utc)
        logging.info(f"Storing manual prediction with timestamp: {timestamp.strftime('%Y-%m-%d %I:%M %p %Z')}")
        readings_collection.insert_one({
            "username": current_user.username,
            "nitrogen": N,
            "phosphorus": P,
            "potassium": K,
            "temperature": temp,
            "humidity": humidity,
            "ph": ph,
            "crop": crop or "Unknown",
            "reason": reason,
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
        
        # Convert UTC timestamps to EAT before rendering
        eat = pytz.timezone('Africa/Nairobi')
        for reading in readings:
            if reading['timestamp'].tzinfo is None:
                reading['timestamp'] = pytz.utc.localize(reading['timestamp'])
            reading['timestamp'] = reading['timestamp'].astimezone(eat)
        
        return render_template(
            'index.html',
            result=result,
            crop=crop,
            image_file=image_file,
            reason=reason,
            readings=readings,
            username=current_user.username,
            current_page=page,
            total_pages=total_pages
        )


    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        flash(f"Prediction error: {str(e)}", "error")
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
