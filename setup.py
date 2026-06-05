import platform

from setuptools import find_packages, setup

install_requires = [
    "pytest",
    "torchinfo",
    "tqdm",
    "scikit-learn",
    "matplotlib",
    "ray[data,train,tune,serve]",
    "hyperopt",
    "optuna",
    "hpbandster",
    "ConfigSpace",
    "bayesian-optimization",
    "ax-platform",
    "safetensors",
]
print(f"Platform: {platform.system()}")

if platform.system() == "Linux":
    print(f"Installing PyTorch with GPU support for {platform.system()}")
    # Install PyTorch with GPU support
    install_requires.append("torch")
    install_requires.append("torchvision")
    install_requires.append("torchaudio")
elif platform.system() == "Darwin":
    # Install PyTorch for Mac
    install_requires.append("torch")
    install_requires.append("torchvision")
    install_requires.append("torchaudio")
else:
    raise ValueError(f"Unsupported platform: {platform.system()}")

setup(
    name="ptmelt",
    version="0.1.4",
    description="PyTorch Machine Learning Toolbox (PT-MELT)",
    url="https://github.com/NREL/pt-melt",
    author="Nicholas T. Wimer",
    author_email="nwimer@nrel.gov",
    license="BSD 3-Clause License",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        # "Programming Language :: Python :: 3.11",
    ],
    # python_requires=">=3.8, <3.12",
)
