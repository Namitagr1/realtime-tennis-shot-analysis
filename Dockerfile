# Use lightweight Python base
FROM python:3.11-slim

# System deps for OpenCV + ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Expose Streamlit port
EXPOSE 8080

# Default run
CMD ["streamlit", "run", "src/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
