from adet.config import get_cfg

def setup_cfg(backbone="R_50"):
    cfg = get_cfg()
    
    if backbone == "R_50":
        config_file = r"C:\Users\hanta\Documents\projects\Tread_Detect\configs\R_50\tread_detect_R50.yaml"
    elif backbone == "ViT":
        config_file = r"C:\Users\hanta\Documents\projects\Tread_Detect\configs\ViTAEv2_S\tread_detect_ViT.yaml"
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Available options are 'R_50' and 'ViT'.")

    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg
