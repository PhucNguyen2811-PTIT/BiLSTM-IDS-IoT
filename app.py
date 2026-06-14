import numpy as np
import tensorflow as tf
import time
import requests  # Thư viện để gửi Webhook
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Reshape, Bidirectional, LSTM
from tensorflow.keras.models import Model
from fastapi import FastAPI, BackgroundTasks
import uvicorn
import os

# Tối ưu RAM cho Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# --- Định nghĩa Model ---
def get_model():
    inputs = Input(shape=(13,))
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 
    x = Reshape((4, 16))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(LSTM(32, dropout=0.2))(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(5, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

model = get_model()
model.load_weights('bilstm_botiot.weights.h5')

# Biến toàn cục để chống spam mail
last_alert_time = 0 

# --- Hàm gửi Webhook (Thay cho SMTP) ---
def send_alert_email(attack_type, confidence):
    webhook_url = "https://hook.eu1.make.com/soz31pnrhct1eulhdz7nd5bqx41wqgir"
    payload = {"attack": attack_type, "conf": str(confidence)}
    try:
        response = requests.post(webhook_url, json=payload)
        print(f"Webhook phản hồi: {response.status_code}")
    except Exception as e:
        print(f"Lỗi gửi Webhook: {e}")

# --- Route Test cho bạn ---
@app.get("/test-mail-debug")
async def test_mail_debug(background_tasks: BackgroundTasks):
    background_tasks.add_task(send_alert_email, "TEST_ATTACK", 0.99)
    return {"status": "success", "message": "Đã kích hoạt gửi test qua Webhook!"}

# --- Logic dự đoán chính ---
@app.post("/predict")
async def predict(data: dict, background_tasks: BackgroundTasks):
    global last_alert_time
    
    data_min = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    data_max = np.array([262212.0, 2.496762990951538, 100.0, 4.980471134185791, 11.0, 
                         4.981882095336914, 100.0, 58823.52734375, 383726.34375, 
                         4.999999046325684, 4.0, 65535.0, 65535.0])
    
    features = np.array(data['features']).reshape(1, 13)
    scaled_features = (features - data_min) / (data_max - data_min)
    
    prediction = model.predict(scaled_features, verbose=0)
    
    label_idx = np.argmax(prediction)
    CLASS_NAMES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']
    result = CLASS_NAMES[label_idx]
    conf = float(np.max(prediction))
    
    # Logic cảnh báo
    if result != 'Normal' and conf > 0.3:
        if (time.time() - last_alert_time) > 3:
            background_tasks.add_task(send_alert_email, result, conf)
            last_alert_time = time.time()
    
    status = "safe" if result == "Normal" else "alert"
    return {"status": status, "type": result, "confidence": conf}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
