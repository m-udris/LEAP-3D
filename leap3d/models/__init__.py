import enum

from leap3d.models.base import BaseModel
from leap3d.models.cnn import CNN
from leap3d.models.unet2d import UNet2D

from leap3d.models.lightning import LEAP3D_CNN, LEAP3D_UNet2D

class Architecture(enum.Enum):
    LEAP3D_CNN = 'leap3d_cnn'
    LEAP3D_UNET2D = 'leap3d_unet2d'

    def get_model(self, *arg, **kwargs) -> BaseModel:
        if self == Architecture.LEAP3D_CNN:
            return LEAP3D_CNN(*arg, **kwargs)
        if self == Architecture.LEAP3D_UNET2D:
            return LEAP3D_UNet2D(*arg, **kwargs)
        raise ValueError(f'Unknown architecture: {self}')
