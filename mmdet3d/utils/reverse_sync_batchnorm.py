import mmcv
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d


# Fallback for BatchNormXd
def _BatchNormXd(num_features, eps, momentum, affine, track_running_stats):
    # Decide which dimensional BN to use (can be adapted if needed)
    return nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

def revert_sync_batchnorm(module):
    """Convert all `SyncBatchNorm` and `mmcv.ops.SyncBatchNorm` layers to `BatchNormXd`.

    Args:
        module (nn.Module): The module containing `SyncBatchNorm` layers.

    Returns:
        module_output: The converted module with `BatchNormXd` layers.
    """
    module_output = module
    module_checklist = [
        torch.nn.SyncBatchNorm,
        torch.nn.modules.batchnorm.SyncBatchNorm,  # in case someone uses this
    ]
    if hasattr(mmcv, 'ops') and hasattr(mmcv.ops, 'SyncBatchNorm'):
        module_checklist.append(mmcv.ops.SyncBatchNorm)

    if isinstance(module, tuple(module_checklist)):
        module_output = _BatchNormXd(module.num_features, module.eps,
                                     module.momentum, module.affine,
                                     module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight.copy_(module.weight)
                module_output.bias.copy_(module.bias)
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))

    del module
    return module_output
