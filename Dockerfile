# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the repository's code into the container
COPY . /app

# Install dependencies from requirements.txt (if present)
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run the application
CMD ["python", "app.py"]
