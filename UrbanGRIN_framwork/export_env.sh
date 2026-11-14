#!/bin/bash
set -u
cd "$(dirname "$0")" || exit 1

python -m pip list --format=freeze 2>/dev/null | grep -v -E "^-e |@ file://" | sort -f > requirements_full.txt

python - <<'EOF' > requirements.txt
import importlib.metadata as md
keys = ["torch", "torchvision", "transformers", "trl", "peft", "accelerate", "datasets",
        "qwen-vl-utils", "numpy", "pillow", "scipy", "safetensors", "tokenizers", "huggingface-hub"]
for k in keys:
    try:
        print(f"{k}=={md.version(k)}")
    except md.PackageNotFoundError:
        pass
EOF

{
  echo "# Environment"
  echo
  echo "- Container: \`$(basename "${APPTAINER_CONTAINER:-${SINGULARITY_CONTAINER:-none}}")\`"
  echo "- OS: $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"
  echo "- Python: $(python -c 'import sys; print(sys.version.split()[0])')"
  python - <<'EOF'
import torch
print(f"- PyTorch: {torch.__version__}")
print(f"- ROCm/HIP: {getattr(torch.version, 'hip', None)}  CUDA: {torch.version.cuda}")
EOF
  echo "- Hardware used: 1 node, 8 GCDs (4x AMD MI250X), 64 GB per GCD"
  echo
  echo "Key packages are pinned in \`requirements.txt\`; the full list is in \`requirements_full.txt\`."
} > ENVIRONMENT.md

cat ENVIRONMENT.md; echo; cat requirements.txt
