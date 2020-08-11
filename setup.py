import setuptools


with open("README.md") as f:
    readmefile_contents = f.read()

setuptools.setup(
    name = "plda",
    version = "0.1.0",
    author = "Ravi B. Sojitra",
    author_email="ravisoji@stanford.edu",
    description = 
        "Probabilistic Linear Discriminant Analysis & Classification",
    long_description = readmefile_contents,
    long_description_content_type = "text/markdown",
    url = "https://github.com/RaviSoji/plda",
    packages  =[
        "plda"
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires = '>=3.5'
)
