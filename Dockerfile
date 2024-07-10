FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    nano \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

# Open the terminal
CMD ["python train.py 0 1 /workspace/datasets/tless/xyz_data"]
