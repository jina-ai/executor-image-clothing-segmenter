import os
import torch
import torchvision.transforms as transforms

from collections import OrderedDict


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    return model


def load_checkpoint_mgpu(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model


class NormalizeImage:
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean: float, std: float):
        """
        Initialization
        """
        assert isinstance(mean, float)
        if isinstance(mean, float):
            self.mean = mean
        if isinstance(std, float):
            self.std = std
        self._normalize_1 = transforms.Normalize(self.mean, self.std)
        self._normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self._normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor: torch.Tensor):
        """
        Callable object
        """
        if image_tensor.shape[0] == 1:
            return self._normalize_1(image_tensor)
        elif image_tensor.shape[0] == 3:
            return self._normalize_3(image_tensor)
        elif image_tensor.shape[0] == 18:
            return self._normalize_18(image_tensor)
        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"
