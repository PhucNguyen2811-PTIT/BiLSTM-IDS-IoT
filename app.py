import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Reshape, Bidirectional, LSTM
from tensorflow.keras.models import Model
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# 1. Định nghĩa Min/Max để chuẩn hóa dữ liệu (Cực kỳ quan trọng)
data_min = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
data_max = np.array([262212.0, 2.496762990951538, 100.0, 4.980471134185791, 11.0, 
                     4.981882095336914, 100.0, 58823.52734375, 383726.34375, 
                     4.999999046325684, 4.0, 65535.0, 65535.0])

# 2. Khai báo khung model (Phải y hệt lúc train)
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
    outputs = Dense(5, activation='softmax')(x) # Vẫn để 5 đầu ra vì model cũ đã train 5 loại
    return Model(inputs=inputs, outputs=outputs)

# 3. Load trọng số
model = get_model()
model.load_weights('bilstm_botiot.weights.h5')
print("Model đã nạp trọng số thành công!")

# 4. Danh sách nhãn (Bỏ Normal)
CLASS_NAMES = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']

@app.post("/predict")
async def predict(data: dict):
    # Chuẩn hóa dữ liệu thô từ client
    features = np.array(data['features']).reshape(1, 13)
    scaled_features = (features - data_min) / (data_max - data_min)
    
    # Dự đoán
    prediction = model.predict(scaled_features, verbose=0)
    label_idx = np.argmax(prediction)
    result = CLASS_NAMES[label_idx]
    
    return {
        "status": "attack",
        "type": result,
        "confidence": float(np.max(prediction))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
