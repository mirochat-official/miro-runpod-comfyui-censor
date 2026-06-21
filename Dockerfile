FROM runpod/worker-comfyui:5.7.1-sdxl

RUN python -m pip uninstall -y onnxruntime onnxruntime-gpu || true

RUN python -m pip install --no-cache-dir \
    pillow \
    numpy \
    opencv-python-headless \
    "onnxruntime-gpu==1.20.1"

COPY censor.py /censor.py
COPY patch_handler.py /patch_handler.py

RUN python /patch_handler.py

ENV CENSOR_ENABLED=true
ENV CENSOR_MODEL_PATH=/runpod-volume/models/censor/nsfw-anime-medium-x1280.onnx
ENV CENSOR_CLASSES=penis,vagina,anus
ENV CENSOR_CONF=0.20
ENV CENSOR_IOU=0.45
ENV CENSOR_PADDING=0.35
ENV CENSOR_FAIL_OPEN=false
ENV CENSOR_IMGSZ=1280
