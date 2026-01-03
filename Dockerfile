FROM python:3.14-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
