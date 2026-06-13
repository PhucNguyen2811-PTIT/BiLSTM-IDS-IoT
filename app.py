import numpy as np
import tensorflow as tf
import time
from fastapi import FastAPI

app = FastAPI()

# 1. Định nghĩa Min/Max
data_min = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
data_max = np.array([262212.0, 2.496762990951538, 100.0, 4.980471134185791, 11.0, 
                     4.981882095336914, 100.0, 58823.52734375, 383726.34375, 
                     4.999999046325684, 4.0, 65535.0, 65535.0])

# 2. Load model trực tiếp từ file .keras (Không cần định nghĩa kiến trúc bằng code nữa)
try:
    # compile=False giúp load nhanh hơn và tránh lỗi liên quan đến loss function khi train
    model = tf.keras.models.load_model('bilstm_botiot.keras', compile=False)
    model.trainable = False 
    print("Model đã nạp thành công từ file .keras!")
except Exception as e:
    print(f"Lỗi nạp model: {e}")

CLASS_NAMES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']

@app.post("/predict")
async def predict(data: dict):
    start_time = time.perf_counter()
    
    # Chuẩn hóa dữ liệu
    features = np.array(data['features']).reshape(1, 13)
    scaled_features = (features - data_min) / (data_max - data_min)
    
    # Dự đoán (training=False bắt buộc cho Inference)
    prediction = model(scaled_features, training=False).numpy()
    
    label_idx = np.argmax(prediction)
    result = CLASS_NAMES[label_idx]
    
    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000
    
    return {
        "status": "attack" if result != 'Normal' else "safe",
        "type": result,
        "confidence": float(np.max(prediction)),
        "latency_ms": round(processing_time_ms, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)