# Use an official Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
