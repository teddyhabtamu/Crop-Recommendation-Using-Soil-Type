import json
import pymongo
from datetime import datetime
import time

# MongoDB Atlas connection
mongo_uri = "mongodb+srv://tewodroshabtamu29:1234tttt@cluster0.oxizy.mongodb.net/SensorData"
client = pymongo.MongoClient(mongo_uri)
db = client["SensorData"]
collection = db["Readings"]

print("Connected to MongoDB Atlas")

try:
    # Simulate reading from file instead of serial
    with open("serial_data.txt", "r") as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    data["timestamp"] = datetime.utcnow()
                    collection.insert_one(data)
                    print(f"Inserted: {data}")
                except json.JSONDecodeError:
                    print("Error decoding JSON:", line)
                except Exception as e:
                    print("Error:", e)
            time.sleep(2)  # Simulate 2-second delay
except KeyboardInterrupt:
    print("Exiting...")
finally:
    client.close()