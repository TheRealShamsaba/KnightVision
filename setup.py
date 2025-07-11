

from setuptools import setup, find_packages

setup(
    name="knightvision",
    version="0.1.0",
    packages=find_packages(include=["ai", "bot", "core", "data_utils", "scripts"]),
    install_requires=[
        "torch==2.2.0",
        "torchvision==0.17.0",
        "numpy==1.26.0",
        "requests==2.31.0",
        "python-chess==1.999",
        "zstandard==0.22.0",
        "python-dotenv==1.0.0",
        "tensorboard==2.17.0",
        "tensorflow==2.17.1",
        "pygame==2.5.2",
    ],
)