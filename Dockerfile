# ===== Base Image (CUDA + Python) =====
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ===== System Dependencies =====
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git wget curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ===== Workdir =====
WORKDIR /app

# ===== Copy project files =====
COPY . /app

# ===== Install Python dependencies =====
RUN pip3 install --upgrade pip \
 && pip3 install --no-cache-dir -r requirements.txt

# ===== Gradio Port =====
EXPOSE 7860

# ===== Run App =====
CMD ["python3", "app.py"]
