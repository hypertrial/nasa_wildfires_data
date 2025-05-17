from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="wildfire_analysis",
    version="0.1.0",
    author="Wildfire Analysis Team",
    author_email="wildfire@example.com",
    description="Spatial analysis of wildfire risk by integrating satellite fire detections with meteorological data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wildfire_analysis",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "pyarrow>=8.0.0",
        "matplotlib>=3.5.0",
        "numba>=0.56.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "wildfire-analysis=wildfire_analysis.main:main",
        ],
    },
) 