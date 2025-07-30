# Use a slim and stable Python base image
FROM --platform=linux/amd64 python:3.10

# Set the working directory 
WORKDIR /app


COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py .
COPY src/ ./src/

# Copy the model, input directory
COPY models/ ./models/
COPY app/input/ ./app/input/

# run when the container starts
CMD ["python", "main.py"]
