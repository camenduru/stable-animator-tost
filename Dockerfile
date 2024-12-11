FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d /content -o cuda_12.6.2_560.35.03_linux.run && sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post3 && \
    pip install opencv-contrib-python imageio imageio-ffmpeg ffmpeg-python av runpod && \
    pip install diffusers transformers accelerate timm decord einops scipy pandas coloredlogs flatbuffers numpy && \
    pip install packaging protobuf sympy imageio-ffmpeg insightface facexlib opencv-python-headless gradio onnxruntime-gpu && \
    git clone https://github.com/Francis-Rings/StableAnimator /content/StableAnimator && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/Animation/face_encoder.pth -d /content/StableAnimator/checkpoints/Animation -o face_encoder.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/Animation/pose_net.pth -d /content/StableAnimator/checkpoints/Animation -o pose_net.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/Animation/unet.pth -d /content/StableAnimator/checkpoints/Animation -o unet.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/DWPose/dw-ll_ucoco_384.onnx -d /content/StableAnimator/checkpoints/DWPose -o dw-ll_ucoco_384.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/DWPose/yolox_l.onnx -d /content/StableAnimator/checkpoints/DWPose -o yolox_l.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/models/antelopev2/1k3d68.onnx -d /content/StableAnimator/checkpoints/models/antelopev2 -o 1k3d68.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/models/antelopev2/2d106det.onnx -d /content/StableAnimator/checkpoints/models/antelopev2 -o 2d106det.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/models/antelopev2/genderage.onnx -d /content/StableAnimator/checkpoints/models/antelopev2 -o genderage.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/models/antelopev2/glintr100.onnx -d /content/StableAnimator/checkpoints/models/antelopev2 -o glintr100.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/models/antelopev2/scrfd_10g_bnkps.onnx -d /content/StableAnimator/checkpoints/models/antelopev2 -o scrfd_10g_bnkps.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/raw/main/stable-video-diffusion-img2vid-xt/model_index.json -d /content/StableAnimator/checkpoints/stable-video-diffusion-img2vid-xt -o model_index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/raw/main/stable-video-diffusion-img2vid-xt/feature_extractor/preprocessor_config.json -d /content/StableAnimator/checkpoints/stable-video-diffusion-img2vid-xt/feature_extractor -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/raw/main/stable-video-diffusion-img2vid-xt/image_encoder/config.json -d /content/StableAnimator/checkpoints/stable-video-diffusion-img2vid-xt/image_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/stable-video-diffusion-img2vid-xt/image_encoder/model.safetensors -d /content/StableAnimator/checkpoints/stable-video-diffusion-img2vid-xt/image_encoder -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/raw/main/stable-video-diffusion-img2vid-xt/scheduler/scheduler_config.json -d /content/StableAnimator/checkpoints/stable-video-diffusion-img2vid-xt/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/raw/main/stable-video-diffusion-img2vid-xt/unet/config.json -d /content/StableAnimator/checkpoints/stable-video-diffusion-img2vid-xt/unet -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/stable-video-diffusion-img2vid-xt/unet/diffusion_pytorch_model.safetensors -d /content/StableAnimator/checkpoints/stable-video-diffusion-img2vid-xt/unet -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/raw/main/stable-video-diffusion-img2vid-xt/vae/config.json -d /content/StableAnimator/checkpoints/stable-video-diffusion-img2vid-xt/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/stable-video-diffusion-img2vid-xt/vae/diffusion_pytorch_model.safetensors -d /content/StableAnimator/checkpoints/stable-video-diffusion-img2vid-xt/vae -o diffusion_pytorch_model.safetensors && \
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/stable-video-diffusion-img2vid-xt/svd_xt.safetensors -d /content/StableAnimator/checkpoints/stable-video-diffusion-img2vid-xt -o stable-video-diffusion-img2vid-xt_xt.safetensors && \
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/stable-video-diffusion-img2vid-xt/svd_xt_image_decoder.safetensors -d /content/StableAnimator/checkpoints/stable-video-diffusion-img2vid-xt -o stable-video-diffusion-img2vid-xt_xt_image_decoder.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/FrancisRing/StableAnimator/resolve/main/inference.zip -d /content/StableAnimator/checkpoints -o inference.zip && cd /content/StableAnimator && unzip /content/StableAnimator/checkpoints/inference.zip
    
COPY ./worker_runpod.py /content/StableAnimator/worker_runpod.py
WORKDIR /content/StableAnimator
CMD python worker_runpod.py