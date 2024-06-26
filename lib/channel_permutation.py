import torch


def calculate_heuristic_channel_permutation(pruning_metric: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Score each channel based on the sum of the importance scores in the channel as indicated by the ``pruning_metric``
    and return a permutation of the channels that sorts channels in the ascending order of channel scores.  See 
    Figure 3 in https://www.preprints.org/manuscript/202310.1487/v1.
    """
    channel_score = pruning_metric.sum(dim=dim)
    return torch.argsort(channel_score)


def register_input_channel_permutation_hook(module: torch.nn.Module, channel_permutation: torch.Tensor):
                    
    def permute_input(module, inp):
        assert len(inp) == 1
        inp = inp[0]
        assert inp.shape[-1] == len(channel_permutation), (
            f"input: {inp.shape} cp: {len(channel_permutation)} weight: {module.weight.data.shape}"
        )
        channel_permutation_on_device = channel_permutation.to(inp.device)
        inp = inp[..., channel_permutation_on_device]
        return inp
                    
    handle = module.register_forward_pre_hook(permute_input)
    return handle
