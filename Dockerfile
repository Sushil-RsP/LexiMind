# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install flask numpy pdfplumber PyPDF2 sentence-transformers gdown joblib

# Expose port (HF Spaces expects 7860)
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
