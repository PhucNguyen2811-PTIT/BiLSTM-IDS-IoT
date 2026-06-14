import numpy as np
import tensorflow as tf
import time
import os
from fastapi import FastAPI

app = FastAPI()

# 1. Định nghĩa Min/Max
data_min = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
data_max = np.array([262212.0, 2.496762990951538, 100.0, 4.980471134185791, 11.0, 
                     4.981882095336914, 100.0, 58823.52734375, 383726.34375, 
                     4.999999046325684, 4.0, 65535.0, 65535.0])

# 2. Load model toàn cục
print("Đang khởi tạo model...")
model = tf.keras.models.load_model('bilstm_botiot.keras', compile=False)
model.trainable = False
# Quan trọng: "Làm nóng" model để tránh bị treo ở request đầu tiên
model._make_predict_function() 
print("Model đã sẵn sàng!")

CLASS_NAMES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']

@app.post("/predict")
async def predict(data: dict):
    # Dự đoán không dùng training=False nếu không cần thiết để tiết kiệm CPU
    features = np.array(data['features']).reshape(1, 13)
    scaled_features = (features - data_min) / (data_max - data_min)
    
    # Dự đoán
    prediction = model.predict(scaled_features, verbose=0)
    
    label_idx = np.argmax(prediction)
    result = CLASS_NAMES[label_idx]
    
    return {
        "status": "attack" if result != 'Normal' else "safe",
        "type": result,
        "confidence": float(np.max(prediction))
    }
