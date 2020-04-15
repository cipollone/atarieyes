# flake8: noqa
# Because this is processed with Black

from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    dependency_links=["git+https://github.com/cipollone/keras-rl.git@tf2#egg=keras-rl"],
    install_requires=["opencv-python", "numpy", "gym[atari]", "tensorflow==2.1"],
    name="atarieyes",
    version="0.0.4",
    author="Roberto Cipollone",
    author_email="cipollone.rt@gmail.com",
    description="RL on the Atari Games with feature extraction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cipollone/atarieyes",
    packages=find_namespace_packages(include=["atarieyes*"]),
    classifiers=["Programming Language :: Python :: 3",],
    python_requires="~=3.6",
)
