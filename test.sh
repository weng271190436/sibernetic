#!/bin/bash
set -ex

ruff format *.py tests/*.py
ruff check *.py tests/*.py

make clean
make

./run_all_tests.sh $@

