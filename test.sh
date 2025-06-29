#!/bin/bash
set -ex

./setup.sh

ruff format *.py
ruff check *.py

make clean
make

./run_all_tests.sh

