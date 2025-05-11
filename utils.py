import torch


def get_available_gpus():
    if torch.cuda.is_available():
        return [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
    else:
        return []