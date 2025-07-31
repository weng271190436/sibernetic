#!/bin/bash
set -ex

ruff format *.py wcon/*.py
ruff check *.py wcon/*.py

make clean
make

./run_all_tests.sh $@

