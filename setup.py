from setuptools import find_packages, setup


setup(
    name="pretraining",
    version="0.0.1",
    description="PyTorch FSDP pretraining workshop",
    packages=find_packages(),
    install_requires=["torch >= 2.1"],
)
