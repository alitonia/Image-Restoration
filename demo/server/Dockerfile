FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

WORKDIR /app

# Copy training requirements and install
# Note: We use the requirements from the training folder
COPY training/requirements.txt ./training/
RUN pip install --no-cache-dir -r training/requirements.txt

# Copy server requirements and install
COPY demo/server/package*.json ./demo/server/
WORKDIR /app/demo/server
RUN npm install

# Copy source code
WORKDIR /app
COPY training/src ./training/src
COPY training/scripts/inference.py ./training/scripts/
COPY training/weights ./training/weights
COPY demo/server ./demo/server

WORKDIR /app/demo/server

# Environment variables for the Node server to find Python and its dependencies
ENV PYTHON_PATH=python3
ENV INFERENCE_SCRIPT_PATH=/app/training/scripts/inference.py
ENV PORT=5000

# Create uploads and results directories
RUN mkdir -p uploads results

EXPOSE 5000

CMD ["node", "index.js"]
