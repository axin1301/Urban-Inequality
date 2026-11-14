import os

BASE = os.environ.get("CITYRIUI_BASE", "/path/to/CityRiUI")

DATA_ROOTS = {
    "GC":  os.path.join(BASE, "UI-COT-data/CoT_and_Original_Data/CoT_Data"),
    "DR":  os.path.join(BASE, "UI-COT-data/dr/CoT"),
    "RIR": os.path.join(BASE, "UI-COT-data/bw/CoT"),
}

if os.environ.get("CITYRIUI_DATA_DIR"):
    _d = os.environ["CITYRIUI_DATA_DIR"]
    DATA_ROOTS = {t: os.path.join(_d, t) for t in ("GC", "DR", "RIR")}

IMAGE_ROOT = os.path.join(BASE, "UI-COT-data/all_images/UI-COT-images_full")

MODEL_NAME = os.environ.get("CITYRIUI_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
HF_CACHE   = os.path.join(BASE, "hf_cache")
OUTPUT_DIR = os.environ.get("CITYRIUI_OUTPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))

SIF  = (
    os.environ.get("CITYRIUI_SIF", "/path/to/container.sif")
)
VENV = os.path.join(BASE, "grpo_env")

TRAIN_CFG = dict(
    max_sv_images                 = int(os.environ.get("CITYRIUI_MAX_SV", 10)),
    num_generations               = 4,
    learning_rate                 = 1e-6,
    per_device_train_batch_size   = 4,
    gradient_accumulation_steps   = 1,

    max_steps                     = -1,
    num_train_epochs              = 1.0,

    max_new_tokens                = 640,

    warmup_steps                  = 50,

    save_steps                    = 20,
    logging_steps                 = 10,
)

SLURM = dict(
    account   = "<your_slurm_account>",
    partition = "standard-g",
    nodes     = 1,
    gpus      = 8,
)
