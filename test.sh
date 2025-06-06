#!/bin/bash
set -ex

ruff format *.py
ruff check *.py

make clean
make

./run_all_tests.sh

