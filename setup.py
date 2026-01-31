from setuptools import setup, find_packages
import os

# Remove LICENSE file if it exists to prevent setuptools from adding it
if os.path.exists('LICENSE'):
    os.rename('LICENSE', 'LICENSE.bak')

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setup(
        name="beamz",
        version="0.1.20",
        author="Quentin Wach",
        author_email="quentin.wach+beamz@gmail.com",
        description="EM package to create inverse / generative designs for your photonic devices with ease and efficiency.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/QuentinWach/beamz",
        packages=find_packages(),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Physics",
        ],
        python_requires=">=3.8",
        install_requires=[
            "numpy>=1.24.4",
            "matplotlib>=3.7.5",
            "gdspy>=1.6.0",
            "scipy>=1.13.0",
            "rich>=13.9.4",
            "shapely>=2.0.6",
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "optax>=0.1.0"
        ],
        extras_require={
            "dev": [
                "pytest>=7.0.0",
                "pytest-cov>=4.0.0",
                "black>=22.0.0",
                "isort>=5.0.0",
                "flake8>=4.0.0",
                "myst-parser>=2.0.0"
            ],
            "gpu": [
                "torch>=2.6.0",
            ],
        },
        include_package_data=True,
    )
finally:
    # Restore the LICENSE file
    if os.path.exists('LICENSE.bak'):
        os.rename('LICENSE.bak', 'LICENSE') 