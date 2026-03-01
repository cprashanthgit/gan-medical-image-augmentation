"""
Setup script for GAN-based Data Augmentation package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gan-medical-augmentation",
    version="1.0.0",
    author="Prash",
    author_email="prash@example.com",
    description="GAN-based Data Augmentation for Medical Image Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gan-medical-augmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-dcgan=train_dcgan:main",
            "train-stylegan2=train_stylegan2:main",
            "evaluate=evaluate:main",
        ],
    },
)
