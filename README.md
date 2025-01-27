# Smart IoT-Based Crop Selection System Using Soil Type Analysis

## **Overview**
This project integrates IoT and AI technologies to recommend suitable crops based on soil and environmental parameters. By collecting real-time data from IoT sensors and processing it through machine learning models, the system helps optimize agricultural productivity and resource efficiency.

---

## **Features**
- Real-time data collection using IoT sensors.
- Preprocessing and normalization of sensor data.
- Machine learning model for crop recommendation (Random Forest Classifier).
- Flask-based web interface for user-friendly crop suggestions.

---

## **Technologies Used**

### **Hardware**
- ESP32 Microcontroller
- DHT22 (Temperature and Humidity Sensor)
- NPK Sensor
- pH Sensor
- Rainfall Sensor (Optional or Simulated)

### **Software**
- **Programming Languages**: Python, HTML, CSS, JavaScript
- **Frameworks**: Flask (for web development)
- **Libraries**: 
  - Data processing: Pandas, NumPy
  - Machine Learning: Scikit-learn
  - Visualization: Matplotlib, Seaborn
  - IoT Simulation: Wokwi (if applicable)

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/Crop-Recommendation-Using-Soil-Type
.git
cd Crop-Recommendation-Using-Soil-Type
```

### **2. Install Dependencies**
Ensure you have Python installed (>=3.8). Install the required Python packages:
```bash
pip install -r requirements.txt
```

### **3. Connect IoT Sensors**
- Connect the sensors (NPK, pH, DHT22) to the ESP32 microcontroller.
- Upload the Arduino code for data transmission to the ESP32.

### **4. Data Preparation**
- Place the `Crop_recommendation.csv` dataset in the root directory.
- Ensure data is clean and ready for processing.

---

## **Code Workflow**

### **1. Data Loading and Preprocessing**
- Load the dataset using Pandas.
- Handle missing values and normalize features (MinMaxScaler and StandardScaler).
- Encode crop labels numerically for compatibility with the model.

### **2. Model Training**
- Train the Random Forest Classifier on soil and environmental data.
- Perform hyperparameter tuning for optimal accuracy.
- Save the trained model using `pickle` for reuse.

### **3. Flask Web Interface**
- Create a user-friendly web interface to:
  - Accept sensor inputs (Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, rainfall).
  - Display recommended crops based on model predictions.
- Start the Flask server using:
```bash
python app.py
```
- Access the web app at `http://127.0.0.1:5000`.

---

## **File Structure**
```
├── Crop_recommendation.csv       # Dataset file
├── model.pkl                     # Trained machine learning model
├── minmaxscaler.pkl              # MinMaxScaler object
├── standscaler.pkl               # StandardScaler object
├── app.py                        # Flask application
├── templates/                    # HTML templates for Flask
│   └── index.html                # Main web page
├── static/                       # Static files (CSS, images, etc.)
│   └── plant.png                 # Example icon
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## **How to Use**
1. Run the Flask app:
   ```bash
   python app.py
   ```
2. Enter soil and environmental data in the web interface.
3. View the crop recommendation based on the provided data.

---

## **Future Enhancements**
- Integrate real-time cloud data storage.
- Add support for additional sensors (e.g., salinity, multi-spectral imaging).

---

## **Contributors**
- Tewodros Habtamu
- Wondwosen Abera
- Advisor: Tsegamlak Terefe (PhD)

---

## **License**
This project is licensed under the MIT License.
