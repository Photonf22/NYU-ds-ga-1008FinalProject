from copy import deepcopy

from wejepa.backbones import adapt_config_for_backbone
from wejepa.config import hf224_config

# start from the Hugging Face-friendly defaults and adapt to a ViT backbone
cfg = adapt_config_for_backbone(deepcopy(hf224_config()), "vit_b_16")
