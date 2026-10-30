from typing import Optional

import torch
from torch.utils.data import Subset

import timm

from dengine.models.decorators import register_model
from dengine.models.utils import get_unique_targets, get_image_channels
from dengine.models import ModuleBase


@register_model()
class TimmModelBase(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        model_name: str,
        *args,
        in_chans: Optional[int] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        num_channels = in_chans or get_image_channels(dataset)
        num_classes = num_classes or len(get_unique_targets(dataset))

        timm_kwargs = {k[8:]: v for k, v in kwargs.items() if k.startswith("__timm__")}
        self._model = timm.create_model(
            model_name,
            num_classes=num_classes,
            in_chans=num_channels,
            **timm_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


def create_model(model_name: str, **kwargs):
    return {
        "client.local_model.target": TimmModelBase,
        "client.local_model.arguments.model_name": model_name,
        **{f"client.local_model.arguments.__timm__{key}": value for key, value in kwargs.items()}
    }
