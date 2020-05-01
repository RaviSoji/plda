import setuptools

with open("README.md") as f:
    readmefile_contents = f.read()

setuptools.setup(
    name="plda",
    version="0.1.0",
    description="Probabilistic Linear Discriminant Analysis & classification",
    author="Ravi Sojitra",
    long_description=readmefile_contents,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    python_requires=">= 3.5.*",
    packages=[
        "plda",
    ],
)
