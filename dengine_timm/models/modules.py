from typing import Optional

import torch
from torch.utils.data import Subset

import timm

from dengine.models.decorators import register_model
from dengine.models.utils import get_unique_targets, get_image_channels, get_image_size
from dengine.models import ModuleBase


@register_model()
class TimmModelBase(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        model_name: str,
        *args,
        image_size: Optional[int] = None,
        in_chans: Optional[int] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        image_size = image_size or get_image_size(dataset)
        num_channels = in_chans or get_image_channels(dataset)
        num_classes = num_classes or len(get_unique_targets(dataset))

        self._model = timm.create_model(
            model_name,
            image_size=image_size,
            num_classes=num_classes,
            num_channels=num_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


def create_model(model_name: str, **kwargs):
    return {
        "client.target": TimmModelBase.__name__,
        **{f"client.arguments.{key}": value for key, value in kwargs.items()}
    }
