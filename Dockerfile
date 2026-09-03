# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
RUN apt update && apt install -y python3 python3-pip ffmpeg libsm6 libxext6
RUN apt update && apt install -y git cmake ninja-build build-essential \
	libboost-program-options-dev libboost-graph-dev libboost-system-dev \
	libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev \
	libgtest-dev libgmock-dev libsqlite3-dev libglew-dev qtbase5-dev \
	libqt5opengl5-dev libcgal-dev libceres-dev libxrandr-dev \
	libxinerama-dev libxcursor-dev libxi-dev libglfw3-dev

ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;8.9;9.0"
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /tmp/
RUN pip install setuptools cython wheel
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install opencv-python joblib tqdm plyfile matplotlib wandb

COPY ./submodules/diff-gaussian-rasterization ./diff-gaussian-rasterization
COPY ./submodules/fused-ssim ./fused-ssim
COPY ./submodules/simple-knn ./simple-knn
RUN pip install ./diff-gaussian-rasterization
RUN pip install ./fused-ssim
RUN pip install ./simple-knn
RUN rm -rf ./diff-gaussian-rasterization
RUN rm -rf ./fused-ssim
RUN rm -rf ./simple-knn
RUN pip install tensorboard
COPY ./submodules/diff-gaussian-rasterization-trunc ./diff-gaussian-rasterization-trunc
RUN pip install ./diff-gaussian-rasterization-trunc
RUN rm -rf ./diff-gaussian-rasterization-trunc

WORKDIR /home/

