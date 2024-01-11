from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ATGraFS-ext",
    version="0.0.1",
    packages=find_packages(),
    install_requires=install_requires,
    author="Orlando Villegas",
    author_email="ovillegas.bello0317@gmail.com",
    description='Automatic MOF generator.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ovillegasb/ATGraFS-ext',
    python_requires=">=3.12.0",
)
