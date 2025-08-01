import sys
import os

if os.getcwd().endswith("tests"):
    os.chdir("..")

sys.path.append(".")

from sibernetic_c302 import run

sim_dir, reportj = run(
    noc302=True,
    duration=3,
    logstep=10,
    configuration="worm_alone_half_resolution",
    simName="test_worm_alone_half",  # Explicitly set simulation name
)

print("TEST: Saved simulation to: %s" % sim_dir)
