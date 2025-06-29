#!/bin/bash
set -ex

# Install system dependencies for building Sibernetic and the PyTorch solver
apt-get update
apt-get install -y python3-dev ocl-icd-opencl-dev libglu1-mesa-dev freeglut3-dev libglew-dev

# Install Python packages
pip install torch ruff
