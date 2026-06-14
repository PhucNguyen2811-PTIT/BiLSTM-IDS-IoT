import numpy as np
import tensorflow as tf
import time
import os
import smtplib
from email.message import EmailMessage
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Reshape, Bidirectional, LSTM
from tensorflow.keras.models import Model
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# --- Cấu hình Email ---
EMAIL_SENDER = "nguyenhoangphuc2811@gmail.com"
EMAIL_PASSWORD = "tzma ifbt pvwt gdfw" 
EMAIL_RECEIVER = "nguyenhoangphuc2811@gmail.com"
last_alert_time = 0 # Biến để chống spam mail

# --- Định nghĩa Min/Max chuẩn hóa ---
data_min = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
data_max = np.array([262212.0, 2.496762990951538, 100.0, 4.980471134185791, 11.0, 
                     4.981882095336914, 100.0, 58823.52734375, 383726.34375, 
                     4.999999046325684, 4.0, 65535.0, 65535.0])

# --- Khung Model ---
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
print("Model đã nạp trọng số thành công!")

# --- Hàm gửi mail ---
def send_alert_email(attack_type, confidence):
    msg = EmailMessage()
    msg['Subject'] = f"CẢNH BÁO: Phát hiện tấn công {attack_type}!"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content(f"Hệ thống IDS phát hiện tấn công!\n\nLoại: {attack_type}\nĐộ tin cậy: {confidence:.2f}\nThời gian: {time.ctime()}")
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)

@app.post("/predict")
async def predict(data: dict):
    global last_alert_time
    start_model = time.perf_counter()
    
    features = np.array(data['features']).reshape(1, 13)
    scaled_features = (features - data_min) / (data_max - data_min)
    
    prediction = model.predict(scaled_features, verbose=0)
    prediction[0][2] = 0 # Ép Normal về 0
    
    label_idx = np.argmax(prediction)
    MAPPED_NAMES = ['DDoS', 'DoS', 'Unknown', 'Reconnaissance', 'Theft']
    result = MAPPED_NAMES[label_idx]
    conf = float(np.max(prediction))
    
    # Gửi mail nếu phát hiện tấn công và cách lần gửi trước > 300s
    if result != 'Unknown' and conf > 0.0:
        if (time.time() - last_alert_time) > 300:
            try:
                send_alert_email(result, conf)
                last_alert_time = time.time()
            except Exception as e:
                print(f"Lỗi gửi mail: {e}")
    
    latency_ms = (time.perf_counter() - start_model) * 1000
    return {"status": "attack", "type": result, "confidence": conf, "latency_ms": round(latency_ms, 2)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
