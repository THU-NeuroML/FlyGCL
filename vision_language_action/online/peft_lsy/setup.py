from setuptools import find_packages, setup


setup(
    name="peft",
    version="0.17.1.dev0",
    description="Local PEFT runtime package for FlyGCL experiments",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={
        "peft": [
            "py.typed",
            "tuners/boft/fbd/fbd_cuda.cpp",
            "tuners/boft/fbd/fbd_cuda_kernel.cu",
        ]
    },
    python_requires=">=3.9.0",
    install_requires=[
        "numpy>=1.17",
        "packaging>=20.0",
        "psutil",
        "pyyaml",
        "torch>=1.13.0",
        "transformers",
        "tqdm",
        "accelerate>=0.21.0",
        "safetensors",
        "huggingface_hub>=0.25.0",
    ],
)
