from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import pybind11

ext_modules = [
    Pybind11Extension(
        "quant_cpp",
        [
            "src/cpp/technical_indicators.cpp",
            "src/cpp/statistical_models.cpp",
            "src/cpp/option_pricing.cpp",
            "src/cpp/python_bindings.cpp",
        ],
        include_dirs=[
            pybind11.get_cmake_dir(),
        ],
        cxx_std=17,
    ),
]

setup(
    name="quant-trading-platform",
    version="1.0.0",
    author="Quantitative Trading Team",
    description="Advanced quantitative trading and forecasting platform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "yfinance>=0.1.70",
        "pybind11>=2.9.0",
        "PyYAML>=6.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    entry_points={
        "console_scripts": [
            "quant-trading=main_engine:main",
        ],
    },
)
