from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    dependency_links=[
        "git+https://github.com/tensorforce/tensorforce.git@0.5.4#egg=tensorforce"
    ],
    install_requires=["opencv-python", "numpy", "gym[atari]", "tensorflow==2.1"],
    name="atarieyes",
    version="0.0.1",
    author="Roberto Cipollone",
    author_email="cipollone.rt@gmail.com",
    description="Feature extraction for Atari Games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cipollone/atarieyes",
    packages=setuptools.find_packages(include=["atarieyes"]),
    classifiers=["Programming Language :: Python :: 3",],
    python_requires="~=3.6",
)
