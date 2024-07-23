import torch
import torch.nn.functional as F
from adet.utils.misc import inverse_sigmoid 

def sigmoid_focal_loss(inputs, targets, num_inst, alpha: float = 0.25, gamma: float = 2): 
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss.ndim == 4:
        return loss.mean((1, 2)).sum() / num_inst
    elif loss.ndim == 3:
        return loss.mean(1).sum() / num_inst
    else:
        raise NotImplementedError(f"Unsupported dim {loss.ndim}")


def prepare_for_dn(dn_args, tgt_weight, embedweight, batch_size, training, num_queries, voc_size, hidden_dim, texts_enc, num_points):
    """
    The major difference from DN-DAB-DETR is that the author process pattern embedding pattern embedding in its detector
    forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, scalar, label_noise_scale, box_noise_scale, num_patterns
    :param tgt_weight: use learnbal tgt in dab deformable detr
    :param embedweight: positional anchor queries
    :param batch_size: bs
    :param training: if it is training or inference
    :param num_queries: number of queires
    :param voc_size: number of classes
    :param hidden_dim: transformer hidden dim
    :param texts_enc: encode labels in dn
    :return:
    """

    if training:
        targets, scalar, texts_noise_scale, ctrl_points_noise_scale, num_patterns, contrastive = dn_args  # noise
    # ---------------------------
    else:
        num_patterns = dn_args 
    # -------------------------------------------

    if num_patterns == 0: 
        num_patterns = 1

    # tgt = None 
    # refpoint_emb = None

    # for truncation
    n_i = 120

    if training: 
        if contrastive:
            new_targets = []  # for contrastive 
            for t in targets:
                new_t = {} 
                # Handle situations where the number of targets is too large
                if t['labels'].shape[0] <= n_i:
                    new_t['texts'] = torch.cat([t['texts'], torch.full((t['texts'].shape[0], num_points), voc_size, dtype=torch.int64).cuda()], dim=0)    
                    new_t['ctrl_points'] = torch.cat([t['ctrl_points'], t['ctrl_points']], dim=0)  
                    new_t['labels'] = torch.cat([t['labels'], torch.tensor(len(t['labels']) * [voc_size], dtype=torch.int64).cuda()], dim=0)
                    new_t['bd_points'] = torch.cat([t['bd_points'], t['bd_points']], dim=0)
                    new_t['beziers'] = torch.cat([t['beziers'], t['beziers']], dim=0)
                    # -----------------------------------------------------------------------
                    new_t['beziers_pts'] = torch.cat([t['beziers_pts'], t['beziers_pts']], dim=0)
                    # -----------------------------------------------------------------------
                    new_targets.append(new_t)
                
                else:
                    new_t['texts'] = torch.cat([t['texts'][:n_i], torch.full((t['texts'][:n_i].shape[0], num_points), voc_size, dtype=torch.int64).cuda()], dim=0)    
                    new_t['ctrl_points'] = torch.cat([t['ctrl_points'][:n_i], t['ctrl_points'][:n_i]], dim=0)  
                    new_t['labels'] = torch.cat([t['labels'][:n_i], torch.tensor(len(t['labels'][:n_i]) * [voc_size], dtype=torch.int64).cuda()], dim=0)
                    new_t['bd_points'] = torch.cat([t['bd_points'][:n_i], t['bd_points'][:n_i]], dim=0)
                    new_t['beziers'] = torch.cat([t['beziers'][:n_i], t['beziers'][:n_i]], dim=0)
                    # -----------------------------------------------------------------------
                    new_t['beziers_pts'] = torch.cat([t['beziers_pts'][:n_i], t['beziers_pts'][:n_i]], dim=0)
                    # -----------------------------------------------------------------------
                    new_targets.append(new_t) 

            targets = new_targets 

        known = [(torch.ones_like(t['texts'])).cuda() for t in targets] # [ [ 1, 1], [1, 1, 1], ... ]
        known = [k.view(-1) for k in known] 
        known_idx = [torch.nonzero(t) for t in known]
        known_num = [sum(k) for k in known]

        count = int(max(known_num))

        # to use fix number of dn queries
        min_num = scalar
        if count <= num_points or count >= num_points * n_i:
            scalar = 1
        else:
            scalar = min(min_num, n_i // (count//num_points))

        batch_idx = torch.cat([torch.full_like(t['texts'].long(), i) for i, t in enumerate(targets)])  
        batch_idx = batch_idx.view(-1)  

        batch_idx_bezier = torch.cat([torch.full_like(t['labels'].unsqueeze(-1).repeat(1, 4).long(), i) for i, t in enumerate(targets)])
        batch_idx_bezier = batch_idx_bezier.view(-1)

        unmask_ctrl_points = unmask_texts = torch.cat(known)  

        # Take texts and ctrl_points from the target
        texts = torch.cat([t['texts'] for t in targets], dim=0) 
        labels = torch.cat([t['labels'] for t in targets], dim=0)
        bd_points = torch.cat([t['bd_points'] for t in targets], dim=0)
        beziers_points = torch.cat([t['beziers'] for t in targets], dim=0)

        beziers_points_pts = torch.cat([t['beziers_pts'] for t in targets], dim=0)  # for compute distance
        beziers_points_pts = beziers_points_pts[:, :4, :] # for distance

        # for text embedding
        # texts_for_embedding = texts.clone()
        texts_for_embedding = character_sliding(texts, voc_size)   

        ctrl_points = torch.cat([t['ctrl_points'] for t in targets], dim=0) 

        # for noise 
        distances = beziers_points_pts[:, :, :] - beziers_points[:, :, :]
        distances = distances.reshape(-1, 2).repeat(scalar, 1)  

        known_indice = torch.nonzero(unmask_ctrl_points + unmask_texts) 
        known_indice = known_indice.view(-1)

        # for add noise
        known_indice = known_indice.repeat(scalar, 1).view(-1)  
        known_bid = batch_idx.repeat(scalar, 1).view(-1) 
        known_bid_beziers = batch_idx_bezier.repeat(scalar, 1).view(-1) 

        known_texts = texts.repeat(scalar, 1).view(-1) 

        known_labels = labels.repeat(scalar, 1).view(-1)
        known_bd_points = bd_points.repeat(scalar, 1, 1).view(-1, 4)

        known_texts_for_embedding = texts_for_embedding.repeat(scalar, 1).view(-1)

        known_ctrl_points = ctrl_points.repeat(scalar, 1, 1).view(-1, 2)  # ctrl points repeated five times 
        known_bezier_points = beziers_points.repeat(scalar, 1, 1).view(-1, 2) 

        known_texts_expand = known_texts_for_embedding  
        known_bezier_points_expand = known_bezier_points.clone()  

        # noise on the texts
        if texts_noise_scale > 0:
            p = torch.rand_like(known_texts_expand.float())  # # Uniform distribution between 0 and 1, used for interpolating noise
            chosen_indice = torch.nonzero(p < (texts_noise_scale)).view(-1)  # usually half of bbox noise
            new_texts = torch.randint_like(chosen_indice, 0, voc_size + 1)  # randomly put a new one here
            known_texts_expand.scatter_(0, chosen_indice, new_texts)
            
        if ctrl_points_noise_scale > 0:
            assert known_bezier_points.shape[-1] == 2
            diff = distances

            if contrastive: 
                rand_sign = torch.randint_like(known_bezier_points_expand, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0 
                rand_part = torch.rand_like(known_bezier_points_expand)  # 0-1 Evenly distributed
               
                positive_idx = torch.tensor(range(len(beziers_points)//2 * beziers_points.shape[1])).long().cuda().unsqueeze(0).repeat(scalar, 1)  
                positive_idx += (torch.tensor(range(scalar)) * len(beziers_points) * beziers_points.shape[1]).long().cuda().unsqueeze(1)  # 

                positive_idx = positive_idx.flatten()  # Select the coordinates of positive
                negative_idx = positive_idx + len(beziers_points)//2 * beziers_points.shape[1] # Select the coordinates of negative

                rand_part[negative_idx] += 1.0  # negative: value + 1
                rand_part *= rand_sign  

                known_bezier_points_expand += torch.mul(rand_part, diff).cuda() * ctrl_points_noise_scale  
                known_bezier_points_expand = known_bezier_points_expand.clamp(min=0.0, max=1.0)  # point coordinates are restricted to 0-1
            else: 
                assert NotImplementedError 

        if contrastive:
            known_texts_expand.scatter_(0, negative_idx, voc_size) 
            p = torch.rand_like(known_texts_expand.float()) 
            chosen_indice = torch.nonzero(p < (texts_noise_scale / 2)).view(-1)  
            new_texts = torch.randint_like(chosen_indice, 0, voc_size + 1)  # randomly put a new one here
            known_texts_expand.scatter_(0, chosen_indice, new_texts)

        m = known_texts_expand.long().to('cuda')
        input_texts_embed = texts_enc(m) 

        # add dn part indicator
        indicator1 = torch.ones([input_texts_embed.shape[0], 1]).cuda() 
        input_texts_embed = torch.cat([input_texts_embed, indicator1], dim=1)
        input_bezier_points_embed = inverse_sigmoid(known_bezier_points_expand) 
        
        known_num_bezier = [x//texts.shape[1]*beziers_points.shape[1] for x in known_num] 
        single_pad = int(max(known_num))  
        single_pad_bezier = int(max(known_num_bezier)) 

        pad_size = int(single_pad * scalar) 
        pad_size_bezier = int(single_pad_bezier * scalar)

        padding_texts = torch.zeros(pad_size, hidden_dim).cuda()    
        padding_bezier_points = torch.zeros(pad_size_bezier, 2).cuda()

        input_query_texts = padding_texts.repeat(batch_size, 1, 1) 
        input_bezier_points = padding_bezier_points.repeat(batch_size, 1, 1)  

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):  
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()  
            
            map_known_indice_bezier = torch.cat([torch.tensor(range(num)) for num in known_num_bezier])
            map_known_indice_bezier = torch.cat([map_known_indice_bezier + single_pad_bezier * i for i in range(scalar)]).long()
            
        if len(known_bid):  
            input_query_texts[(known_bid.long(), map_known_indice)] = input_texts_embed 
            input_bezier_points[known_bid_beziers.long(), map_known_indice_bezier] = input_bezier_points_embed
            input_query_texts = input_query_texts.view(batch_size, -1, texts.shape[1], hidden_dim)  
            input_bezier_points = input_bezier_points.view(batch_size, -1, beziers_points.shape[1], 2) 

        # intra self-attention need mask, and inter does not
        output_single_pad = single_pad//texts.shape[1]  
        output_pad_size = pad_size // texts.shape[1]
        tgt_size = output_pad_size + num_queries * num_patterns  

        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0

        attn_mask[output_pad_size:, :output_pad_size] = True 
        # reconstruct cannot see each other 
        for i in range(scalar):
            if i == 0:
                attn_mask[output_single_pad * i:output_single_pad * (i + 1), output_single_pad * (i + 1):output_pad_size] = True
            if i == scalar - 1:
                attn_mask[output_single_pad * i:output_single_pad * (i + 1), :output_single_pad * i] = True
            else:
                attn_mask[output_single_pad * i:output_single_pad * (i + 1), output_single_pad * (i + 1):output_pad_size] = True
                attn_mask[output_single_pad * i:output_single_pad * (i + 1), :output_single_pad * i] = True
       
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_texts_pnts': (known_labels, known_texts, known_ctrl_points, known_bd_points),
            'known_idx': known_idx,
            'pad_size': output_pad_size,
            'scalar': scalar,
            'contrastive' : contrastive,
        }
    
    else:  # no dn for inference
        input_query_texts = None
        input_query_ctrl_points = None
        attn_mask = None
        mask_dict = None

    return input_query_texts, input_bezier_points, attn_mask, mask_dict



def dn_post_process(outputs_class, outputs_texts, outputs_coord, outputs_bd_coord, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    if mask_dict and mask_dict['pad_size'] > 0: 
        outputs_known_class = outputs_class[:, :, :mask_dict['pad_size'], :, :] # [ levels, bs, query size, max_recognition_length, hidden dim] 
        outputs_known_texts = outputs_texts[:, :, :mask_dict['pad_size'], :, :]  # [ levels, bs, query size, max_recognition_length, hidden dim] 
        outputs_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :, :]
        outputs_known_bd_coord = outputs_bd_coord[:, :, :mask_dict['pad_size'], :, :] 

        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :, :]   
        outputs_texts = outputs_texts[:, :, mask_dict['pad_size']:, :, :]  
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :, :] 
        outputs_bd_coord = outputs_bd_coord[:, :, mask_dict['pad_size']:, :, :] 

        mask_dict['output_known_texts_pnts']=(outputs_known_class,outputs_known_texts,outputs_known_coord,outputs_known_bd_coord)
    return outputs_class, outputs_texts, outputs_coord, outputs_bd_coord, mask_dict


def prepare_for_loss(mask_dict):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    Returns:
    """
    output_known_class, output_known_texts, output_known_coord, outputs_known_bd_points = mask_dict['output_known_texts_pnts']
    known_labels, known_texts, known_ctrl_points, known_bd_points = mask_dict['known_texts_pnts']  

    map_known_indice = mask_dict['map_known_indice']  # [0, 1, 2, 3, 4, ...249, 0, ...,99...]  
    # [0, 1, 2, 3, 4, ..., 0, 1, 2, 3, 4, ...]

    known_indice = mask_dict['known_indice']  # [0, 1, 2, 3, 4, ...349, ....]  1750
    # [0, 1, 2, 3, 4, ...]

    batch_idx = mask_dict['batch_idx'] 
    bid = batch_idx[known_indice]
    num_tgt = known_indice.numel() 

    if len(output_known_texts) > 0:
        output_known_class_shape = output_known_class.shape  
        output_known_class = output_known_class.view(output_known_class_shape[0], output_known_class_shape[1], -1, output_known_class_shape[4])
        output_known_class = output_known_class.permute(1, 2, 0, 3)[[bid, map_known_indice]].permute(1, 0, 2) 

        output_known_texts_shape = output_known_texts.shape 
        output_known_texts = output_known_texts.view(output_known_texts_shape[0], output_known_texts_shape[1], -1, output_known_texts_shape[4])
        output_known_texts = output_known_texts.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2) 

        # [ levels, bs, qs, hdim ] -> [ bs, qs, lvls, hdim] -> [ lvls, bs * qs, hdim ]
        output_known_coord_shape = output_known_coord.shape  
        output_known_coord = output_known_coord.view(output_known_coord_shape[0], output_known_coord_shape[1], -1, output_known_coord_shape[4])
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)

        outputs_known_bd_points_shape = outputs_known_bd_points.shape
        outputs_known_bd_points = outputs_known_bd_points.view(outputs_known_bd_points_shape[0], outputs_known_bd_points_shape[1], -1, outputs_known_bd_points_shape[4])
        outputs_known_bd_points = outputs_known_bd_points.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2) 

    if mask_dict['contrastive'] :
        scalar = mask_dict['scalar']  # 5
        num_tgt = num_tgt // 2  
        num_ctrl_points = num_tgt // scalar 

        positive_idx = torch.tensor(range(num_ctrl_points)).long().cuda().unsqueeze(0).repeat(scalar, 1)  
        positive_idx += (torch.tensor(range(scalar)) * num_ctrl_points * 2).long().cuda().unsqueeze(1)

        positive_idx = positive_idx.flatten()

        output_known_coord = output_known_coord[:, positive_idx, :]  
        output_known_coord = output_known_coord.view(output_known_coord_shape[0], -1, output_known_coord_shape[3], output_known_coord_shape[4])  

        outputs_known_bd_points = outputs_known_bd_points[:, positive_idx, :]
        outputs_known_bd_points = outputs_known_bd_points.view(outputs_known_bd_points_shape[0], -1, outputs_known_bd_points_shape[3], outputs_known_bd_points_shape[4])


        known_ctrl_points = known_ctrl_points[positive_idx,:]  
        known_ctrl_points = known_ctrl_points.view(-1, output_known_coord_shape[3], output_known_coord_shape[4]) 

        known_bd_points = known_bd_points[positive_idx, :]
        known_bd_points = known_bd_points.view(-1, outputs_known_bd_points_shape[3], outputs_known_bd_points_shape[4]) 

        known_texts = known_texts.view(-1, output_known_texts_shape[-2]) 
        output_known_texts = output_known_texts.view(output_known_texts_shape[0], -1, output_known_texts_shape[3],output_known_texts_shape[4]) 

        output_known_class = output_known_class.view(output_known_class_shape[0], -1, output_known_class_shape[3], output_known_class_shape[4])

    return known_labels, known_texts, known_ctrl_points,known_bd_points, output_known_class, output_known_texts, output_known_coord, outputs_known_bd_points, num_tgt

def tgt_loss_labels(src_labels_, tgt_labels_, num_inst): 
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        num_inst = num_inst // src_labels_.shape[1]  # num_inst // 25
        src_labels, tgt_labels = src_labels_.unsqueeze(0), tgt_labels_.unsqueeze(0) 
        shape = list(src_labels.squeeze(-1).shape) 
        # shape[-1] += 1
        idx = torch.nonzero(tgt_labels==0, as_tuple=True)

        target_classes_onehot = torch.zeros(shape, # 1 70 25
                                        dtype=src_labels.dtype, layout=src_labels.layout, device=src_labels.device)
        
        target_classes_onehot[idx] = 1

        target_classes_onehot = target_classes_onehot.unsqueeze(-1)
        loss_ce = sigmoid_focal_loss(src_labels, target_classes_onehot, num_inst,  
                                     alpha=0.25, gamma=2) * src_labels.shape[1]
        
        losses = {'tgt_loss_ce': loss_ce}

        # if log:
        #     # this should probably be a separate loss, not hacked in this one here
        #     losses['tgt_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses
    

# 70 25 38，，   70 25
def tgt_loss_texts(src_texts, tgt_texts, num_tgt, scalar):  
        # CTC loss for classification of points
        if len(tgt_texts) == 0:
            return {
                'tgt_loss_texts_pos': torch.as_tensor(0.).to('cuda'),
                'tgt_loss_texts_neg': torch.as_tensor(0.).to('cuda'),
                # 'tgt_class_error': torch.as_tensor(0.).to('cuda'),
            }
        positive_idx = torch.tensor(range(len(tgt_texts) // scalar // 2)).long().cuda().unsqueeze(0).repeat(scalar, 1)  # 5 5
        l = positive_idx.size(-1)
        positive_idx += (torch.tensor(range(scalar)) * l * 2).long().cuda().unsqueeze(1)  # 

        positive_idx = positive_idx.flatten()  # Select the coordinates of positive
        negative_idx = positive_idx + len(tgt_texts) // scalar // 2  # Select the coordinates of negative

        voc_size = src_texts.size(-1) - 1  

        src_texts_pos = src_texts[positive_idx]  
        src_texts_neg = src_texts[negative_idx]  
        tgt_texts_pos = tgt_texts[positive_idx]
        tgt_texts_neg = tgt_texts[negative_idx] 

        src_texts_pos = src_texts_pos.permute(1, 0, 2)  
        src = F.log_softmax(src_texts_pos, dim=-1)  # shape: (length, n, voc_size+1)

        input_lengths = torch.full((src.size(1),), src.size(0), dtype=torch.long) 
        tgt_lengths = (tgt_texts_pos != voc_size).long().sum(dim=-1)
        tgt_texts_pos = torch.cat([t[:l] for t, l in zip(tgt_texts_pos, tgt_lengths)])
        loss_texts_pos = F.ctc_loss(
                src, 
                tgt_texts_pos, 
                input_lengths,
                tgt_lengths,
                blank=voc_size,
                zero_infinity=True
            )
        
        loss_texts_neg = F.cross_entropy(src_texts_neg.transpose(1, 2), tgt_texts_neg.long())
        # losses = {'tgt_loss_texts_pos': loss_texts_pos}
        losses = {'tgt_loss_texts_pos': loss_texts_pos, 'tgt_loss_texts_neg': loss_texts_neg}
        return losses



def tgt_loss_ctrl_points(src_ctrl_points, tgt_ctrl_points, num_tgt):
        """Compute the L1 regression loss
        """
        num_tgt = num_tgt // 25
        loss_ctrl_points = F.l1_loss(src_ctrl_points, tgt_ctrl_points, reduction='sum')
        losses = {'tgt_loss_ctrl_points': loss_ctrl_points / num_tgt}
        return losses


def tgt_loss_bd_points(src_bd_points, tgt_bd_points, num_inst):  
        num_inst = num_inst // 25 
        loss_bd_points = F.l1_loss(src_bd_points, tgt_bd_points, reduction='sum')
        losses = {'tgt_loss_bd_points': loss_bd_points / num_inst}
        return losses


def compute_dn_loss(mask_dict, training, aux_num, focal_alpha):
    """
       compute dn loss in criterion
       Args:
           mask_dict: a dict for dn information
           training: training or inference flag
           aux_num: aux loss number
           focal_alpha:  for focal loss
       """
    losses = {}
    if training and 'output_known_texts_pnts' in mask_dict:
        """
        known_labels: 70
        known_texts  # 70 25
        known_ctrl_points # 35 25 2
        output_known_class [6, 50, 25, 1]
        output_known_texts  # 6 35 25 38
        output_known_coord  # 6 35 25 2
        """
        known_labels, known_texts, known_ctrl_points, known_bd_points, output_known_class, output_known_texts, output_known_coord, output_known_bd_points, \
        num_tgt = prepare_for_loss(mask_dict)
        scalar = mask_dict['scalar']

        losses.update(tgt_loss_labels(output_known_class[-1], known_labels, num_tgt))
        losses.update(tgt_loss_texts(output_known_texts[-1], known_texts, num_tgt, scalar))
        losses.update(tgt_loss_ctrl_points(output_known_coord[-1], known_ctrl_points, num_tgt))
        losses.update(tgt_loss_bd_points(output_known_bd_points[-1], known_bd_points, num_tgt))

    else:
        losses['tgt_loss_texts_pos'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_texts_neg'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_ctrl_points'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_class_error'] = torch.as_tensor(0.).to('cuda')


    if aux_num:
        for i in range(aux_num):
            # dn aux loss
            if training and 'output_known_texts_pnts' in mask_dict:
                l_dict = tgt_loss_labels(output_known_class[i], known_labels, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

                l_dict = tgt_loss_texts(output_known_texts[i], known_texts, num_tgt, scalar)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

                l_dict = tgt_loss_ctrl_points(output_known_coord[i], known_ctrl_points, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

                l_dict = tgt_loss_bd_points(output_known_bd_points[i], known_bd_points, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

            else:
                l_dict = dict()
                losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
                losses['tgt_loss_texts_pos'] = torch.as_tensor(0.).to('cuda')
                losses['tgt_loss_texts_neg'] = torch.as_tensor(0.).to('cuda')

                losses['tgt_loss_ctrl_points'] = torch.as_tensor(0.).to('cuda')
                losses['tgt_loss_bd_points'] = torch.as_tensor(0.).to('cuda')
                
                # losses['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
    return losses


def character_sliding(input_tensor, padding_value=37):
    input_tensor = input_tensor.clone()

    n = input_tensor.size(1)  
    for idx in range(input_tensor.size(0)):
        n_valid = (input_tensor[idx] != padding_value).sum()  # tensor(7)
        if n_valid == 0:
            continue
        t = n // n_valid.item()  # 25 // 7 = tensor(3)
        k = n % n_valid.item()  # 25 % 7 = tensor(4)

        # first line: [6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 7, 7, 7]
        non_padding_elements = input_tensor[idx][input_tensor[idx] != padding_value]  # 6 5 4 3 2 1 7

        repeat_counts = torch.cat([torch.full((k,), t + 1).cuda(), torch.full((n_valid - k,), t).cuda()])  # tensor([4, 4, 4, 4, 3, 3, 3])
        input_tensor[idx] = torch.repeat_interleave(non_padding_elements, repeat_counts)

    input_tensor = process_continuous_sequence(input_tensor)
    return input_tensor


def process_continuous_sequence(tensor):
    result = torch.zeros_like(tensor)  

    for p in range(tensor.size(0)):
        i = 0
        while i < tensor.size(1):
            start = i
            while i < tensor[p].size(0) - 1 and tensor[p][i] == tensor[p][i + 1]:
                i += 1
            end = i

            if end - start + 1 >= 5:  # If the length of the continuous area is greater than or equal to 5
                num_to_keep = torch.randint(2, 4, (1,))  # 
                indices_to_keep = torch.randint(start, end + 1, (num_to_keep,))  
                result[p][indices_to_keep] = tensor[p][indices_to_keep]

            elif end - start + 1 >= 3:  # If the length of the continuous area is greater than or equal to 3
                num_to_keep = torch.randint(1, 3, (1,))  # 
                indices_to_keep = torch.randint(start, end + 1, (num_to_keep,))  
                result[p][indices_to_keep] = tensor[p][indices_to_keep]

            else:
                num_to_keep = torch.randint(1, 2, (1,))  # Randomly choose to keep one or two numbers
                indices_to_keep = torch.randint(start, end + 1, (num_to_keep,))  
                result[p][indices_to_keep] = tensor[p][indices_to_keep]

            i += 1

    return result


def compute_index_map(n_valid, n_space): 
    n_interval = n_space // n_valid  # 25 // 7 = 3
    return torch.arange(n_valid).cuda() * n_interval  #  torch.arange(n_valid)=tensor([0, 1, 2, 3, 4, 5, 6])  

def insert_data_uniformly(tensor, voc_size):
    result = torch.full_like(tensor, voc_size).cuda()
    for idx in range(tensor.size(0)):  
        n_valid = (tensor[idx] != voc_size).sum()  
        if n_valid > 0: 
            index_map = compute_index_map(n_valid, tensor.size(1))  # tensor([0, 3, 6, 9, 12, 15, 18])
            non_padding_elements = tensor[idx][tensor[idx] != voc_size]  # tensor([6, 5, 4, 3, 2, 1, 7])
            result[idx, index_map] = non_padding_elements 
    return result








