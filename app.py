import numpy as np
import tensorflow as tf
import time  # <--- THIẾU CÁI NÀY
from fastapi import FastAPI

app = FastAPI()

# 1. Định nghĩa Min/Max
data_min = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
data_max = np.array([262212.0, 2.496762990951538, 100.0, 4.980471134185791, 11.0, 
                     4.981882095336914, 100.0, 58823.52734375, 383726.34375, 
                     4.999999046325684, 4.0, 65535.0, 65535.0])

# 2. Focal Loss
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0-eps)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1.0-y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight*ce, axis=1))
    return loss_fn

# 3. Load model
model = tf.keras.models.load_model('bilstm_botiot.h5', custom_objects={'loss_fn': focal_loss()})
CLASS_NAMES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']

@app.post("/predict")
async def predict(data: dict):
    start_time = time.perf_counter()  # <--- BẮT ĐẦU ĐO Ở ĐÂY
    
    features = np.array(data['features'])
    scaled_features = (features - data_min) / (data_max - data_min)
    
    prediction = model.predict(scaled_features.reshape(1, -1), verbose=0) # Thêm verbose=0 để đỡ rác terminal
    label_idx = np.argmax(prediction)
    result = CLASS_NAMES[label_idx]
    
    end_time = time.perf_counter()   # KẾT THÚC ĐO
    processing_time_ms = (end_time - start_time) * 1000 
    
    return {
        "status": "attack" if result != 'Normal' else "safe",
        "type": result,
        "confidence": float(np.max(prediction)),
        "latency_ms": round(processing_time_ms, 2)
    }