from adet.config import get_cfg

def setup_cfg(backbone="R_50"):
    cfg = get_cfg()
    
    if backbone == "R_50":
        config_file = "configs/R_50/TotalText/finetune_150k_tt_mlt_13_15_textocr.yaml"
        opts = ['MODEL.WEIGHTS', 'models/res50_ctw_model_pretrain.pth']
    elif backbone == "ViT":
        config_file = "configs/ViTAEv2_S/TotalText/finetune_150k_tt_mlt_13_15_textocr.yaml"
        opts = ['MODEL.WEIGHTS', 'models/vitaev2_pretrain_tt_model_final.pth']
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Available options are 'R_50' and 'ViT'.")

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg
