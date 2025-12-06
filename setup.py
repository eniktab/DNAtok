"""
DNATok: GPU-accelerated DNA sequence tokenization for genomic foundation models.

Provides 34-244x speedup over standard tokenization through intelligent runtime
optimization, vectorized operations, and GPU acceleration.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dnatokenizer",
    version="0.1.0",
    author="Your Name/Team",
    author_email="your.email@institution.edu",
    description="GPU-accelerated DNA sequence tokenization for genomic foundation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DNATok",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "examples": [
            "transformers>=4.30.0",
            "pandas>=1.5.0",
        ],
    },
    keywords="genomics dna tokenization gpu bioinformatics foundation-models",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/DNATok/issues",
        "Source": "https://github.com/yourusername/DNATok",
        "Documentation": "https://github.com/yourusername/DNATok#readme",
    },
)
