import numpy as np
import tensorflow as tf
import time
import smtplib
from email.message import EmailMessage
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Reshape, Bidirectional, LSTM
from tensorflow.keras.models import Model
from fastapi import FastAPI, BackgroundTasks
import uvicorn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# --- Cấu hình Email ---
EMAIL_SENDER = "nguyenhoangphuc2811@gmail.com"
EMAIL_PASSWORD = "tzmaifbtpvwtgdfw" 
EMAIL_RECEIVER = "nguyenhoangphuc2811@gmail.com"
last_alert_time = 0 

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

# --- Logic xử lý chính ---
@app.post("/predict")
async def predict(data: dict, background_tasks: BackgroundTasks):
    global last_alert_time
    
    # Chuẩn hóa (Min/Max giữ nguyên như cũ)
    data_min = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    data_max = np.array([262212.0, 2.496762990951538, 100.0, 4.980471134185791, 11.0, 
                         4.981882095336914, 100.0, 58823.52734375, 383726.34375, 
                         4.999999046325684, 4.0, 65535.0, 65535.0])
    
    features = np.array(data['features']).reshape(1, 13)
    scaled_features = (features - data_min) / (data_max - data_min)
    
    prediction = model.predict(scaled_features, verbose=0)
    
    # 1. KHÔNG ÉP NORMAL VỀ 0 NỮA -> Để model tự quyết định
    label_idx = np.argmax(prediction)
    CLASS_NAMES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']
    result = CLASS_NAMES[label_idx]
    conf = float(np.max(prediction))
    
    # 2. Logic gửi mail: Chỉ gửi nếu là tấn công (không phải Normal)
    if result != 'Normal' and conf > 0.6:
        if (time.time() - last_alert_time) > 300:
            background_tasks.add_task(send_alert_email, result, conf)
            last_alert_time = time.time()
    
    return {"status": "success", "type": result, "confidence": conf}

def send_alert_email(attack_type, confidence):
    # Hàm này giữ nguyên, nhưng nhớ rằng Render chặn cổng SMTP
    # Bạn sẽ thấy lỗi [Errno 101] trong log, đó là bình thường trên Render Free
    pass
