"""DNAtok: GPU-native tokenisation for genomic foundation models.

Bit-identical to Hugging Face. 2-100x faster on modern single-nucleotide
models (Evo2, NTv3) and supported across the seven major published
genomic foundation model families.
"""
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dnatokenizer",
    version="0.1.0",
    description="GPU-native tokenisation for genomic foundation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"dnatok_bpe_kernel": ["*.cu"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "examples": [
            "transformers>=4.46",
            "biopython>=1.83",
            "pysam>=0.22",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # `dnatok <subcommand>` — plug-and-play CLI shipped with the
            # Docker / Apptainer image. Source: src/dnatok_cli.py.
            "dnatok=dnatok_cli:main",
        ],
    },
    keywords="genomics dna tokenization gpu bioinformatics foundation-models",
)
