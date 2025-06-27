# Quantitative Trading Platform Makefile

# Variables
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fPIC
PYTHON = python3
PIP = pip3

# Directories
SRC_DIR = src
CPP_DIR = $(SRC_DIR)/cpp
PYTHON_DIR = $(SRC_DIR)/python
BUILD_DIR = build
DIST_DIR = dist

# Python binding
PYBIND11_INCLUDES = $(shell python3 -m pybind11 --includes)
PYTHON_SUFFIX = $(shell python3-config --extension-suffix)
MODULE_NAME = quant_cpp$(PYTHON_SUFFIX)

# Source files
CPP_SOURCES = $(wildcard $(CPP_DIR)/*.cpp)
CPP_OBJECTS = $(CPP_SOURCES:$(CPP_DIR)/%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all clean install test setup build-cpp build-python

all: setup build-cpp build-python

# Setup directories
setup:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(DIST_DIR)
	@mkdir -p logs
	@mkdir -p results
	@mkdir -p config

# Build C++ objects
$(BUILD_DIR)/%.o: $(CPP_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) -c $< -o $@

# Build C++ module
build-cpp: $(CPP_OBJECTS)
	$(CXX) -shared $(CXXFLAGS) $(CPP_OBJECTS) -o $(MODULE_NAME)
	@echo "C++ module built successfully: $(MODULE_NAME)"

# Install Python dependencies
install-deps:
	$(PIP) install -r requirements.txt
	@echo "Python dependencies installed"

# Build Python package
build-python: build-cpp
	$(PYTHON) setup.py build_ext --inplace
	@echo "Python package built successfully"

# Install the package
install: build-python
	$(PIP) install -e .
	@echo "Package installed in development mode"

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v --cov=src/python
	@echo "Tests completed"

# Run the main analysis
run-analysis:
	$(PYTHON) $(PYTHON_DIR)/main_engine.py

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(DIST_DIR)
	rm -f $(MODULE_NAME)
	rm -rf *.egg-info
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "Cleaned build artifacts"

# Development setup
dev-setup: setup install-deps build-cpp
	$(PIP) install -e .
	@echo "Development environment setup complete"

# Docker build
docker-build:
	docker build -t quant-trading-platform .
	@echo "Docker image built"

# Docker run
docker-run:
	docker run -v $(PWD)/results:/app/results quant-trading-platform
	@echo "Analysis completed in Docker"

# Format code
format:
	black $(PYTHON_DIR)
	isort $(PYTHON_DIR)
	clang-format -i $(CPP_DIR)/*.cpp $(CPP_DIR)/*.hpp
	@echo "Code formatted"

# Lint code
lint:
	flake8 $(PYTHON_DIR)
	pylint $(PYTHON_DIR)
	@echo "Code linted"

# Generate documentation
docs:
	sphinx-build -b html docs/ docs/_build/
	@echo "Documentation generated"

# Benchmark performance
benchmark:
	$(PYTHON) benchmarks/performance_test.py
	@echo "Benchmark completed"

# Profile code
profile:
	$(PYTHON) -m cProfile -o profile_results.prof $(PYTHON_DIR)/main_engine.py
	@echo "Profiling completed"

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build everything"
	@echo "  setup        - Create necessary directories"
	@echo "  build-cpp    - Build C++ module"
	@echo "  build-python - Build Python package"
	@echo "  install-deps - Install Python dependencies"
	@echo "  install      - Install package in development mode"
	@echo "  test         - Run tests"
	@echo "  run-analysis - Run the main trading analysis"
	@echo "  clean        - Clean build artifacts"
	@echo "  dev-setup    - Setup development environment"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run analysis in Docker"
	@echo "  format       - Format code"
	@echo "  lint         - Lint code"
	@echo "  docs         - Generate documentation"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  profile      - Profile code performance"
	@echo "  help         - Show this help message"
