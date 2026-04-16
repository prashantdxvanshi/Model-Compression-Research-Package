import torch

from .method import PruningMethod
from ..registry import register_method


@register_method('iterative', name='structured_magnitude')
class StructuredMagnitudePruningMethod(PruningMethod):

    def _init(self, target_sparsity=0.5, dim=0):
        self.register_name('original')
        self.register_name('mask')

        self.target_sparsity = target_sparsity
        self.dim = dim  # 0 = neuron (rows), 1 = input features (columns)

        original = getattr(self.module, self.name)
        delattr(self.module, self.name)

        self.module.register_parameter(self.get_name('original'), original)
        self.module.register_buffer(
            self.get_name('mask'),
            torch.ones_like(original, dtype=original.dtype, device=original.device)
        )

    @torch.no_grad()
    def _compute_mask(self):
        weight = self.get_parameters('original')

        # importance (L2 norm)
        importance = torch.norm(weight, dim=self.dim)

        k = int(self.target_sparsity * importance.numel())

        # safety fix (important)
        if k <= 0:
            threshold = importance.min() - 1
        else:
            threshold = torch.kthvalue(importance, k).values

        keep = (importance > threshold)

        if self.dim == 0:
            mask = keep[:, None].expand_as(weight)
        else:
            mask = keep[None, :].expand_as(weight)

        self.set_parameter('mask', mask.to(weight.dtype))

    def masked_weight(self, module):
        original, mask = self.get_parameters('original', 'mask', module=module)
        return original * mask


# OPTIONAL: helper function (same style as library)
def structured_magnitude_pruning(module, name='weight', target_sparsity=0.5, dim=0):
    try:
        method = module.get_pruning_parameters('method', name=name)
        method.target_sparsity = target_sparsity
        method.update_mask(1)
    except AttributeError:
        method = StructuredMagnitudePruningMethod(
            module,
            name,
            target_sparsity=target_sparsity,
            dim=dim
        )
    return module, method
