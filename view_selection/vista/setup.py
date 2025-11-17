# setup.py
from setuptools import setup, find_packages

setup(
    name="voxel_traversal",
    version="0.1.0",
    author="Timothy Chen",
    author_email="chengine@stanford.edu",
    description="GPU-based voxel ray traversal in Pytorch",
    packages=find_packages()
)