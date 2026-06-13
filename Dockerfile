# Sử dụng python slim để giảm dung lượng image
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết (giảm thiểu dung lượng)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào image
COPY . .

# Mở port cho Render
EXPOSE 10000

# Lệnh chạy app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]