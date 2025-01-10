FROM nvcr.io/nvidia/pytorch:21.11-py3

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 wget zip htop screen libgl1-mesa-glx -y

RUN pip install \
    pandas \
    scikit-learn \
    tqdm \
    wandb \
    opencv-python \
    matplotlib \
    seaborn \
    thop
