# Base: NVIDIA NGC PyTorch 26.03 — ships a fresh torch (2.x) + triton, so
# torch.compile works out of the box.
#
# CUDA/driver note: 26.03 is built against CUDA 13.2, which needs a host driver
# from the 595+ branch. With the host driver at 595.97 the container gets CUDA
# natively — no forward-compat layer is needed (and no LD_PRELOAD/compat shims;
# those actively BREAK it by forcing an older userspace libcuda). If you ever
# run this on a host still on the 560 driver, CUDA will be unavailable — either
# update the driver to 595+, or switch the base to 25.06-py3 (CUDA 12.x).
FROM nvcr.io/nvidia/pytorch:26.03-py3

RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    "transformers==4.50.3" \
    "datasets==3.5.0" \
    "huggingface-hub==0.30.1"

RUN python - <<'PY'
import torch, triton, transformers, datasets, huggingface_hub, numba, numpy
print("torch:", torch.__version__, "| triton:", triton.__version__)
print("transformers:", transformers.__version__, "| datasets:", datasets.__version__)
print("numba:", numba.__version__, "| numpy:", numpy.__version__)
assert torch.compile(lambda x: x * 2 + 1) is not None
print("torch.compile: OK")
PY

WORKDIR /workspace

# --- Usage -------------------------------------------------------------------
# Build:
#   docker build -t llm-pretrain-ngc .
#
# GPU check (must be run with --gpus all):
#   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
