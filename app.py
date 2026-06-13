import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dropout, Reshape, Bidirectional, LSTM, Dense, Input
from tensorflow.keras.models import Model
import time
from fastapi import FastAPI

app = FastAPI()

# 1. Định nghĩa Min/Max
data_min = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
data_max = np.array([262212.0, 2.496762990951538, 100.0, 4.980471134185791, 11.0, 
                     4.981882095336914, 100.0, 58823.52734375, 383726.34375, 
                     4.999999046325684, 4.0, 65535.0, 65535.0])

# 2. Định nghĩa kiến trúc khớp 100% với model gốc
def get_model():
    inputs = Input(shape=(13,))
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Reshape((4, 16))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(5, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

# 3. Load model và Weights
model = get_model()
try:
    model.load_weights('bilstm_botiot.weights.h5')
    print("Nạp trọng số thành công!")
except Exception as e:
    print(f"Lỗi nạp trọng số: {e}")

CLASS_NAMES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']

@app.post("/predict")
async def predict(data: dict):
    start_time = time.perf_counter()
    
    features = np.array(data['features']).reshape(1, 13)
    scaled_features = (features - data_min) / (data_max - data_min)
    
    # Dự đoán
    prediction = model(scaled_features, training=False).numpy()
    
    label_idx = np.argmax(prediction)
    result = CLASS_NAMES[label_idx]
    
    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000
    print(f"DEBUG: Nhận được features: {data['features']}")
    return {
        "status": "attack" if result != 'Normal' else "safe",
        "type": result,
        "confidence": float(np.max(prediction)),
        "latency_ms": round(processing_time_ms, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)