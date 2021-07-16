import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agnfinder",
    version="0.0.2",
    author="Maxime Robeyns",
    author_email="maximerobeyns@gmail.com",
    description="Find AGN in XMM/Euclid photometry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3 License",
        "Operating System :: OS Independent",
    ],
)

# walmsleymk1@gmail.com
