# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependency (OpenCV fix)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app ./app
COPY models ./models

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
