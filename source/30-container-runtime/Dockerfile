FROM python:3.12-slim-bookworm@sha256:5d9e6adc9abd558bc0ce05f9ab16cef71328916fc9728f6b81e337af86a737e2

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/cache/hub \
    HOME=/tmp USER=llm LOGNAME=llm PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /opt/llm
COPY requirements.txt ./
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128' \
    && python -m pip install -r requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/llm/image-packages.txt
COPY app/smollm2_serve_selected.py ./
COPY app/smollm2_api_metrics.py ./
COPY app/smollm2_registry.py ./
USER 65532:65532
ENTRYPOINT ["python", "-B", "smollm2_serve_selected.py"]
CMD ["--device", "cuda", "--metrics", "--port", "8003"]
