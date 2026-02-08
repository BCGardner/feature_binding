from setuptools import setup, find_packages


def read_requirements(file_name):
    with open(file_name, "r") as file:
        return [line.strip() for line in file if line.strip()]

setup(
    name='hsnn',
    version='1.0.0',
    author='Brian Gardner',
    author_email="b.gardner@surrey.ac.uk",
    description='Hierarchical SNN simulated using brian2',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://gitlab.surrey.ac.uk/bg0013/feature_binding",
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.11',
    install_requires=read_requirements("requirements.txt")
)
