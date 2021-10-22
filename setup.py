from setuptools import setup

setup(
    name="poly_mt",
    version="1.0",
    packages=["algos", "envs"],
    install_requires=["tqdm", "numpy", "matplotlib", "gym>=0.10", "imageio", "scipy", "torch"],
)
