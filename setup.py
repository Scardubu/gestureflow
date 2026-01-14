“””
GestureFlow setup configuration.
“””
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description

readme_file = Path(**file**).parent / “README.md”
long_description = readme_file.read_text(encoding=“utf-8”) if readme_file.exists() else “”

setup(
name=“gestureflow”,
version=“1.0.0”,
author=“Oscar Ndugbu”,
author_email=“scardubu@gmail.com”,
description=“Production LSTM swipe typing prediction engine”,
long_description=long_description,
long_description_content_type=“text/markdown”,
url=“https://github.com/scardubu/gestureflow”,
project_urls={
“Bug Reports”: “https://github.com/scardubu/gestureflow/issues”,
“Source”: “https://github.com/scardubu/gestureflow”,
“Documentation”: “https://github.com/scardubu/gestureflow#readme”,
“Demo”: “https://gestureflow.scardubu.dev”,
},
packages=find_packages(exclude=[“tests”, “tests.*”, “docs”, “notebooks”]),
classifiers=[
“Development Status :: 4 - Beta”,
“Intended Audience :: Developers”,
“Intended Audience :: Science/Research”,
“Topic :: Scientific/Engineering :: Artificial Intelligence”,
“Topic :: Software Development :: Libraries :: Python Modules”,
“License :: OSI Approved :: MIT License”,
“Programming Language :: Python :: 3”,
“Programming Language :: Python :: 3.9”,
“Programming Language :: Python :: 3.10”,
“Programming Language :: Python :: 3.11”,
“Operating System :: OS Independent”,
],
python_requires=”>=3.9”,
install_requires=[
“torch>=2.1.0”,
“numpy>=1.24.0”,
“pandas>=2.0.0”,
“scikit-learn>=1.3.0”,
“fastapi>=0.104.0”,
“uvicorn[standard]>=0.24.0”,
“pydantic>=2.5.0”,
“scipy>=1.11.0”,
“tqdm>=4.66.0”,
“pyyaml>=6.0.1”,
“requests>=2.31.0”,
],
extras_require={
“dev”: [
“pytest>=7.4.0”,
“pytest-cov>=4.1.0”,
“black>=23.11.0”,
“flake8>=6.1.0”,
“mypy>=1.7.0”,
],
“visualization”: [
“matplotlib>=3.8.0”,
“seaborn>=0.13.0”,
“tensorboard>=2.15.0”,
],
“optimization”: [
“onnx>=1.15.0”,
“onnxruntime>=1.16.0”,
],
“all”: [
“pytest>=7.4.0”,
“pytest-cov>=4.1.0”,
“black>=23.11.0”,
“flake8>=6.1.0”,
“mypy>=1.7.0”,
“matplotlib>=3.8.0”,
“seaborn>=0.13.0”,
“tensorboard>=2.15.0”,
“onnx>=1.15.0”,
“onnxruntime>=1.16.0”,
],
},
entry_points={
“console_scripts”: [
“gestureflow-train=scripts.train_model:main”,
“gestureflow-generate=src.data.generator:main”,
“gestureflow-benchmark=scripts.benchmark:main”,
],
},
keywords=[
“machine-learning”,
“deep-learning”,
“lstm”,
“rnn”,
“nlp”,
“sequence-modeling”,
“gesture-recognition”,
“swipe-typing”,
“pytorch”,
“fastapi”,
],
include_package_data=True,
zip_safe=False,
)