import os
from adet.config import get_cfg

def setup_cfg(backbone="R_50"):
    cfg = get_cfg()
    
    # Get the absolute path of the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if backbone == "R_50":
        config_file = os.path.join(project_root, "configs", "R_50", "tread_detect_R50.yaml")
    elif backbone == "ViT":
        config_file = os.path.join(project_root, "configs", "ViTAEv2_S", "tread_detect_ViT.yaml")
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Available options are 'R_50' and 'ViT'.")

    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg
