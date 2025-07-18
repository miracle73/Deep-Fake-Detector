
"""
Setup script for deepfake detection package
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "An AI model for detecting deepfake images using deep learning techniques."

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "opencv-python>=4.7.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.11.0",
        "pillow>=9.0.0",
        "tqdm>=4.64.0",
        "PyYAML>=6.0",
        "tensorboard>=2.11.0",
        "timm>=0.6.0",
        "albumentations>=1.3.0"
    ]

# Optional dependencies
extras_require = {
    "api": [
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "python-multipart>=0.0.6"
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0"
    ],
    "all": []
}

# Combine all extras
all_deps = []
for deps in extras_require.values():
    all_deps.extend(deps)
extras_require["all"] = list(set(all_deps))

setup(
    name="deepfake-detector",
    version="1.0.0",
    author="AI Developer",
    author_email="developer@example.com",
    description="An AI model for detecting deepfake images using deep learning techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/deepfake-detector",
    project_urls={
        "Bug Tracker": "https://github.com/username/deepfake-detector/issues",
        "Documentation": "https://github.com/username/deepfake-detector/wiki",
        "Source Code": "https://github.com/username/deepfake-detector",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "deepfake_detector": [
            "config/*.yaml",
            "config/*.json",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepfake-train=scripts.train_model:main",
            "deepfake-evaluate=scripts.evaluate_model:main",
            "deepfake-inference=scripts.inference:main",
        ],
    },
    keywords=[
        "deepfake",
        "detection",
        "artificial intelligence",
        "computer vision",
        "deep learning",
        "pytorch",
        "image classification",
        "fake media detection"
    ],
    zip_safe=False,
)