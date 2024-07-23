import numpy as np
import torch
from torch import nn 
import torch.nn.functional as F
from adet.layers.deformable_transformer import DeformableTransformer
from adet.utils.misc import (
    NestedTensor,
    inverse_sigmoid_offset,
    nested_tensor_from_tensor_list, 
    sigmoid_offset
)
from adet.modeling.model.utils import MLP 
from adet.modeling.model.dn_components import prepare_for_dn, dn_post_process, compute_dn_loss
from adet.utils.curve_utils import upcast
from scipy.special import comb as n_over_k



class DETECTION_TRANSFORMER(nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE) 

        self.backbone = backbone

        self.d_model = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.pos_embed_scale = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_classes = 1 # text or not text
        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE  # 37 
        self.sigmoid_offset = False

        self.num_proposals = cfg.MODEL.TRANSFORMER.NUM_QUERIES 
        self.num_points = cfg.MODEL.TRANSFORMER.NUM_POINTS
        self.usedn = cfg.MODEL.USEDN

        # ---------------------------------------------------
        if self.usedn == True:
            self.texts_enc = nn.Embedding(self.voc_size + 1, self.d_model - 1)  # for indicator
        # --------------------------------------------------------------------

        self.transformer = DeformableTransformer(
            temp=cfg.MODEL.TRANSFORMER.TEMPERATURE,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels,
            dec_n_points=self.dec_n_points,
            enc_n_points=self.enc_n_points,
            num_proposals=self.num_proposals,
            num_points=self.num_points,
            usedn=self.usedn
        )

        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            if cfg.MODEL.BACKBONE.NAME == 'build_swin_backbone':
                if cfg.MODEL.SWIN.TYPE == 'tiny' or 'small':
                    num_channels = [192, 384, 768]
                else:
                    raise NotImplementedError
            elif cfg.MODEL.BACKBONE.NAME == 'build_vitaev2_backbone':
                if cfg.MODEL.ViTAEv2.TYPE == 'vitaev2_s':
                    num_channels = [128, 256, 512]
                else:
                    raise NotImplementedError
            else:
                num_channels = [512, 1024, 2048]

            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            num_channels[0], self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                ]
            )
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        # bezier center line proposal after the encoder
        # x_0, y_0, ... , x_3, y_3
        self.bezier_proposal_coord = MLP(self.d_model, self.d_model, 8, 3)
        self.bezier_proposal_class = nn.Linear(self.d_model, self.num_classes)  # text or non-text
        # task specific heads after the decoder
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)  # text or non-text
        self.ctrl_point_text = nn.Linear(self.d_model, self.voc_size + 1)  # specific character class for each point
        self.boundary_head_on = cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD
        if self.boundary_head_on:
            self.boundary_offset = MLP(self.d_model, self.d_model, 4, 3)  # to rebuild the text boundary from queries

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.bezier_proposal_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.ctrl_point_text.bias.data = torch.ones(self.voc_size + 1) * bias_value

        nn.init.constant_(self.bezier_proposal_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.bezier_proposal_coord.layers[-1].bias.data, 0)
        self.transformer.bezier_coord_embed = self.bezier_proposal_coord
        self.transformer.bezier_class_embed = self.bezier_proposal_class

        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)
        if self.boundary_head_on:
            nn.init.constant_(self.boundary_offset.layers[-1].weight.data, 0)
            nn.init.constant_(self.boundary_offset.layers[-1].bias.data, 0)

        ######################################################################
        # shared prediction heads
        ######################################################################
        num_pred = self.num_decoder_layers
        self.ctrl_point_coord = nn.ModuleList(
            [self.ctrl_point_coord for _ in range(num_pred)]
        )
        self.ctrl_point_class = nn.ModuleList(
            [self.ctrl_point_class for _ in range(num_pred)]
        )
        self.ctrl_point_text = nn.ModuleList(
            [self.ctrl_point_text for _ in range(num_pred)]
        )
        if self.boundary_head_on:
            self.boundary_offset = nn.ModuleList(
                [self.boundary_offset for _ in range(num_pred)]
            )

        self.transformer.decoder.ctrl_point_coord = self.ctrl_point_coord

        Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)  
        BezierCoeff = lambda ts: [[Mtk(3, t, k) for k in range(4)] for t in ts]
        curve_token = torch.linspace(0, 1, self.num_points) 
        self.bernstein_matrix = torch.tensor(BezierCoeff(curve_token), requires_grad=False)  # 25 4
        self.to(self.device)

    def init_points_from_bezier_proposals(self, reference_bezier): 
        bz = reference_bezier.shape[0]
        initial_reference_points = reference_bezier.view(bz, reference_bezier.shape[1], 4, 2) 
        initial_reference_points = torch.matmul(
            upcast(self.bernstein_matrix.to(initial_reference_points.device)),
            upcast(initial_reference_points) 
        )
        return initial_reference_points  
    

    def forward(self, samples: NestedTensor, dn_args=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[0]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if dn_args is not None:
            # prepare for dn 
            input_query_texts, bezier_reference_points, attn_mask, mask_dict = \
                prepare_for_dn(dn_args, None, None, src.size(0), self.training, self.num_proposals, self.voc_size,
                            self.d_model, self.texts_enc, self.num_points) 
            if input_query_texts is not None and bezier_reference_points is not None:
                # sometimes the target is empty, add a zero part of label_enc to avoid unused parameters
                input_query_texts += self.texts_enc.weight[0][0]*torch.tensor(0).cuda()
                bezier_reference_points = bezier_reference_points.view(bezier_reference_points.shape[0], bezier_reference_points.shape[1], -1)
                input_query_ctrl_points = self.init_points_from_bezier_proposals(bezier_reference_points) 
                query_embed = torch.cat((input_query_texts, input_query_ctrl_points), dim=-1)
            else:
                query_embed = None  
                mask_dict = None
                attn_mask = None

        else:
            query_embed = None
            mask_dict = None
            attn_mask = None

        # (n_proposal x n_pts, d_model) -> (n_proposal, n_pts, d_model)

        (
            hs,  
            init_reference, 
            inter_references,  
            enc_outputs_class,
            enc_outputs_coord_unact 
        ) = self.transformer(srcs, masks, pos, query_embed, attn_mask, dn_args)

        outputs_texts = []
        outputs_coords = []
        outputs_classes = []
        if self.boundary_head_on:
            outputs_bd_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1] 
            # hs shape: (bs, n_proposal, n_pts, d_model)
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)  
            outputs_class = self.ctrl_point_class[lvl](hs[lvl]) 
            outputs_text = self.ctrl_point_text[lvl](hs[lvl])  # bs, n_proposal, n_pts, voc_size 
            tmp = self.ctrl_point_coord[lvl](hs[lvl])  
            if self.boundary_head_on:  
                tmp_bd = self.boundary_offset[lvl](hs[lvl])  

            if reference.shape[-1] == 2:
                tmp += reference  
                if self.boundary_head_on:
                    tmp_bd += reference.repeat(1, 1, 1, 2)  
            else:
                raise NotImplementedError

            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset) 
            if self.boundary_head_on:
                outputs_bd_coord = sigmoid_offset(tmp_bd, offset=self.sigmoid_offset) 
                outputs_bd_coords.append(outputs_bd_coord)

            outputs_classes.append(outputs_class)
            outputs_texts.append(outputs_text)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes) 
        outputs_text = torch.stack(outputs_texts)  
        outputs_coord = torch.stack(outputs_coords) 
        if self.boundary_head_on:
            outputs_bd_coord = torch.stack(outputs_bd_coords)  
 
        outputs_class, outputs_texts, outputs_coord, outputs_bd_coord, mask_dict = \
            dn_post_process(outputs_class, outputs_text, outputs_coord, outputs_bd_coord, mask_dict)

        out = {
            'pred_logits': outputs_class[-1],  
            'pred_text_logits': outputs_texts[-1], 
            'pred_ctrl_points': outputs_coord[-1], 
            'pred_bd_points': outputs_bd_coord[-1] if self.boundary_head_on else None 
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class,
                outputs_texts,
                outputs_coord,
                outputs_bd_coord if self.boundary_head_on else None
            )

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class,
            'pred_beziers': enc_outputs_coord
        }

        return out, mask_dict

    @torch.jit.unused 
    def _set_aux_loss(self, outputs_class, outputs_text, outputs_coord, outputs_bd_coord):
        if outputs_bd_coord is not None:
            return [
                {'pred_logits': a, 'pred_text_logits': b, 'pred_ctrl_points': c, 'pred_bd_points': d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_text[:-1], outputs_coord[:-1], outputs_bd_coord[:-1])
            ]
        else:
            return [
                {'pred_logits': a, 'pred_text_logits': b, 'pred_ctrl_points': c}
                for a, b, c in zip(outputs_class[:-1], outputs_text[:-1], outputs_coord[:-1])
            ]