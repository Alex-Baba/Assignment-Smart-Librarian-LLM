# Dockerfile

# Start with the base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app

# Expose the port Streamlit will run on
EXPOSE 8501

# Define the command to run the application
CMD ["streamlit", "run", "ui/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]