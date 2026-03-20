# ARA-Net: Atlas-Guided Attention for Alzheimer's Disease Classification
# One-click reproduction environment

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install Python dependencies (torch already in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (exclude large data/checkpoints via .dockerignore if present)
COPY . .

# Default: run full pipeline (requires sample_data/cache_real with preprocessed .npz)
# Override with: docker run ... aranet bash -c "python -m chapter1_foundation.run_experiment_v3 --quick ..."
CMD ["./chapter1_foundation/launch_full_v2.sh"]
