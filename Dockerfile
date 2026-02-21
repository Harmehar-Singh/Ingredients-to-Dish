FROM huggingface/space-pytorch-cpu:1.0.10

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Install your requirements
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
