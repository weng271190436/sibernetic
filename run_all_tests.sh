#This script will run a number of tests using sibernetic_c302.py & check the 
# files produced
set -ex

PYTHON_BIN="python3"
if [[ -x "venv/bin/python" ]]; then
    PYTHON_BIN="venv/bin/python"
fi

# Try to auto-discover c302 checkout if C302_HOME is not already set.
if [[ -z "${C302_HOME:-}" ]] && [[ -d "CElegansNeuroML/CElegans/pythonScripts" ]]; then
    export C302_HOME="$PWD/CElegansNeuroML/CElegans/pythonScripts"
fi

if [[ -n "${C302_HOME:-}" ]]; then
    C302_PATH="$C302_HOME"
    if [[ "$(basename "$C302_PATH")" == "c302" ]]; then
        C302_PATH="$(dirname "$C302_PATH")"
    fi
    export PYTHONPATH="$C302_PATH:./src:${PYTHONPATH:-}"
fi

# No c302
"$PYTHON_BIN" sibernetic_c302.py -test -noc302 -duration 0.1 -simName test_noc302 

# No c302 - test logstep
"$PYTHON_BIN" sibernetic_c302.py -test -noc302 -duration 0.054 -logstep 3 -simName test_noc302_logstep -q

if "$PYTHON_BIN" -c "import c302" >/dev/null 2>&1; then
    # c302
    "$PYTHON_BIN" sibernetic_c302.py -test  -duration 1.1  -c302params C1 -simName test_c302 -q

    # c302 + half_resolution
    "$PYTHON_BIN" sibernetic_c302.py -test  -duration 1 -c302params C0 -configuration worm_alone_half_resolution -simName test_c302_half_resolution -q

    # c302 + TestMuscle 
    "$PYTHON_BIN" sibernetic_c302.py -test -duration 20 -c302params C0 -reference TargetMuscle -configuration worm_alone_half_resolution -logstep 500 -simName test_c302_half_resolution_target_muscle -q
else
    echo "Skipping c302 tests: module 'c302' not found."
    echo "To enable them: clone CElegansNeuroML and set C302_HOME, then ensure PYTHONPATH includes \$C302_HOME:./src"
fi

if [[ ($# -eq 1) && ($1 == '-all') ]]; then

    # Run a simulation with the FW (forward locomotion) c302 configuration with C2 (cond based) cells
    "$PYTHON_BIN" sibernetic_c302.py -test -duration 150 -dt 0.005 -dtNrn 0.05 -logstep 100 -device=CPU -configuration worm_crawl_half_resolution -reference FW -c302params C2 -datareader UpdatedSpreadsheetDataReader2 -simName test_C2_FW-q

    # Run a simulation with the FW (forward locomotion) c302 configuration with W2D (simple passive) cells
    ##python sibernetic_c302.py -test -duration 15.0 -dt 0.005 -dtNrn 0.05 -logstep 100 -device=CPU -configuration worm_crawl_half_resolution -reference FW -c302params W2D -datareader UpdatedSpreadsheetDataReader2

    omv all -V

fi


