#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="wi-compass",
    version="0.1.0",
    description="VQ-VAE based data collection framework for mmWave human pose estimation",
    author="Wi-compass Team",
    python_requires=">=3.11",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=2.0.0",
        "scipy>=1.10.1",
        "tqdm>=4.0.0",
        "PyYAML>=6.0",
        "matplotlib>=3.5.0",
        "h5py>=3.7.0",
    ],

    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "body": [
            "human_body_prior @ git+https://github.com/nghorbani/human_body_prior.git",
        ]
    },

    entry_points={
        "console_scripts": [
            "wi-compass-train=wicompass.train.train_vqvae:main",
            "wi-compass-eval=wicompass.evaluation.scripts.evaluate_model:main",
            "wi-compass-encode=wicompass.evaluation.scripts.encode_dataset:main",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
)
