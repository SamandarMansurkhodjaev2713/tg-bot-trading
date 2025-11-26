from setuptools import setup, find_packages

setup(
    name="educational_ai_finance_project",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "accelerate",
        "peft",
        "joblib",
        "loguru",
    ],
)
