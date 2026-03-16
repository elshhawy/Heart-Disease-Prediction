from setuptools import setup, find_packages

setup(
    name="heart-disease-prediction",
    version="1.0.0",
    description="ML pipeline for heart disease prediction",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "joblib>=1.3",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4"]
    },
)
