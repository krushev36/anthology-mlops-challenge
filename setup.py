from setuptools import find_packages, setup

setup(
    name="mlops",
    version="0.1.0",
    author="Sergio Sanes",
    author_email="sergio.sanes@gmail.com",
    description="A library for MLOps",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/krushev36/mlops",
    packages=find_packages(where="mlops"),
    package_dir={"": "mlops"},
    include_package_data=True,
    install_requires=[
        "pandas>=1.5.3",
        "scikit-learn>=1.2.2",
        "tensorflow>=2.11.0",
        "mlflow>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
