# Use an official Python runtime as the base image
FROM python:3.9-buster as base

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools \
    && pip install -r requirements.txt

# Copy the rest of the application files into the container
COPY . /app/

# Ensure model file is copied
COPY enhanced_bilstm_model.pth /app/

# Set the entrypoint command for Gunicorn to run the Flask app
CMD ["gunicorn", "--workers=4", "--threads=2", "--bind", "0.0.0.0:8000", "app:app"]
