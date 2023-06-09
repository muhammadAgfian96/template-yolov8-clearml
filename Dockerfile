# Ultralytics YOLO 🚀, GPL-3.0 license
# Builds ultralytics/ultralytics:latest image on DockerHub https://hub.docker.com/r/ultralytics/ultralytics
# Image is CUDA-optimized for YOLOv8 single/multi-GPU training and inference

# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# FROM docker.io/pytorch/pytorch:latest
FROM python:3.10.11-slim

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg
# RUN alias python=python3

# Create working directory
WORKDIR /workspace

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel

# RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
# RUN pip install --upgrade git+https://github.com/muhammadAgfian96/ultralytics.git@feat/disable-callbacks#egg=ultralytics
COPY requirements.txt /deps/requirements.txt
RUN pip install -r /deps/requirements.txt


# Set environment variables
ENV OMP_NUM_THREADS=1

# Cleanup
ENV DEBIAN_FRONTEND teletype