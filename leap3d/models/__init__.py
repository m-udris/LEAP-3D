import enum

from leap3d.models.base import BaseModel
from leap3d.models.cnn import CNN
from leap3d.models.unet import UNet, UNet3d, ConditionalUNet, ConditionalUNet3d
from leap3d.models.mlp import MLP
from leap3d.models.cnn import CNN
from leap3d.models.lightning import LEAP3D_UNet2D, LEAP3D_UNet3D, InterpolationMLP

class Architecture(enum.Enum):
    LEAP3D_CNN = 'leap3d_cnn'
    LEAP3D_UNET2D = 'leap3d_unet2d'
    LEAP3D_UNET3D = 'leap3d_unet3d'

    def get_model(self, *args, **kwargs) -> BaseModel:
        if self == Architecture.LEAP3D_UNET2D:
            return LEAP3D_UNet2D(*args, **kwargs)
        if self == Architecture.LEAP3D_UNET3D:
            return LEAP3D_UNet3D(*args, **kwargs)
        raise ValueError(f'Unknown architecture: {self}')
