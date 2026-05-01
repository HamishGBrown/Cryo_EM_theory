from setuptools import setup

setup(
    name="cryo-em-theory",
    version="0.1.0",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "ipywidgets",
        "ipympl",
        "imageio",
        "mrcfile",
        "healpy",
        "pooch",
        "ipykernel",
        "tqdm",
    ],
    extras_require={
        "colab": ["google-colab"],
    },
)
