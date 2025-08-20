from setuptools import setup, find_packages

setup(
    name="video-deepfake-detector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "opencv-python-headless",
        "pillow>=10.1.0",
        "numpy>=1.21.0",
        "google-cloud-aiplatform>=1.35.0",
        "google-cloud-storage>=2.10.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0"
    ],
)
