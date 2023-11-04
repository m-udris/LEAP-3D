from setuptools import setup
from setuptools import find_packages

setup(
    name='leap3d',
    version='0.1.0',
    description='LEAP3D',
    packages=find_packages(include=['leap3d', 'leap3d.*']),
    install_requires=[
        # "pytorch-cuda",
        # "torchaudio",
        # "torchvision",
        # "pytorch",
        # "lightning",
        # "python-dotenv",
        # "wandb",
        # "ipykernel",
        # "matplotlib",
        # "openh264",
        # "scipy",
    ]
)