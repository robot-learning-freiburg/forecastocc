import torch
import torch.nn.functional as F


def resize_rel_pos_bias(pretrained_bias, source_size, target_size):
    # Assume pretrained_bias is [N1, num_heads], target is [N2, num_heads]
    num_heads = pretrained_bias.shape[1]
    patch_bias = pretrained_bias[:-3, :]

    patch_bias = patch_bias.permute(1, 0).reshape(num_heads, 2 * source_size[0] - 1, 2 * source_size[1] - 1)  # [num_heads, 2*SH-1, 2*SW-1]
    # Interpolate
    patch_bias = F.interpolate(patch_bias.unsqueeze(0), size=(2 * target_size[0] - 1, 2 * target_size[1] - 1), mode='bicubic', align_corners=False)
    N2 = (2 * target_size[0] - 1) * (2 * target_size[1] - 1)
    patch_bias = patch_bias.squeeze(0).reshape(num_heads, N2).permute(1, 0)  # [N2, num_heads]
    pretrained_bias = torch.cat([patch_bias, pretrained_bias[-3:, :]], dim=0)  # [N2+3, num_heads]
    return pretrained_bias

def init_checkpoint(checkpoint, model, logger=None, map_location='cpu'):
    assert checkpoint is not None
    assert model is not None
    checkpoint_data = torch.load(checkpoint['path'], map_location=map_location)
    # Hardcoded for copying backbone weights to the model's image backbone
    
    state_dict = checkpoint_data['state_dict'] if 'state_dict' in checkpoint_data else checkpoint_data

    # Create a new dict with updated keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_key = k.replace('backbone.', '', 1)
            if new_key.endswith('.relative_position_bias_table'):
                # Resize the relative position bias table
                new_v = resize_rel_pos_bias(v, (checkpoint['s_h'], checkpoint['s_w']), (checkpoint['t_h'], checkpoint['t_w']))
                v = new_v

        else:
            new_key = k
        new_state_dict[new_key] = v

    if hasattr(model.module,'img_backbone') and hasattr(model.module.img_backbone, 'pos_embed') and model.module.img_backbone.pos_embed is not None and model.module.img_backbone.pos_embed.shape != new_state_dict['pos_embed'].shape:
        len_pos_embed = new_state_dict['pos_embed'].shape[1]
        num_patches = len_pos_embed - 1
        p_s = int(num_patches**0.5)    
        assert p_s * p_s == num_patches, f"num_patches {num_patches} is not a square number"
        src_pos_embed = new_state_dict['pos_embed'][:, 1:, :].reshape(1, p_s, p_s, -1).permute(0, 3, 1, 2)
        src_pos_embed = torch.nn.functional.interpolate(src_pos_embed, size=(model.module.img_backbone.pretrain_size[0] // 16, model.module.img_backbone.pretrain_size[1] // 16), mode='bicubic', align_corners=False)        
        src_pos_embed = src_pos_embed.reshape(1, -1, model.module.img_backbone.pretrain_size[0] // 16 * model.module.img_backbone.pretrain_size[1] // 16).permute(0, 2, 1)
        src_pos_embed = torch.cat([new_state_dict['pos_embed'][:, :1, :], src_pos_embed], dim=1)
        new_state_dict['pos_embed'] = src_pos_embed

    elif hasattr(model.module,'scan_backbone') and hasattr(model.module.scan_backbone, 'pos_embed') and model.module.scan_backbone.pos_embed is not None and model.module.scan_backbone.pos_embed.shape != new_state_dict['pos_embed'].shape:
        len_pos_embed = new_state_dict['pos_embed'].shape[1]
        num_patches = len_pos_embed - 1
        p_s = int(num_patches**0.5)    
        assert p_s * p_s == num_patches, f"num_patches {num_patches} is not a square number"
        src_pos_embed = new_state_dict['pos_embed'][:, 1:, :].reshape(1, p_s, p_s, -1).permute(0, 3, 1, 2)
        src_pos_embed = torch.nn.functional.interpolate(src_pos_embed, size=(model.module.scan_backbone.pretrain_size[0] // 16, model.module.scan_backbone.pretrain_size[1] // 16), mode='bicubic', align_corners=False)        
        src_pos_embed = src_pos_embed.reshape(1, -1, model.module.scan_backbone.pretrain_size[0] // 16 * model.module.scan_backbone.pretrain_size[1] // 16).permute(0, 2, 1)
        src_pos_embed = torch.cat([new_state_dict['pos_embed'][:, :1, :], src_pos_embed], dim=1)
        new_state_dict['pos_embed'] = src_pos_embed


    # Replace the old state_dict
    checkpoint_data['state_dict'] = new_state_dict
    if hasattr(model.module,'img_backbone'):
        model.module.img_backbone.load_state_dict(new_state_dict, strict=False) 
    elif hasattr(model.module,'scan_backbone'):
        model.module.scan_backbone.load_state_dict(new_state_dict, strict=False)    
    logger.info(f'loaded checkpoint from: {checkpoint}')



def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))