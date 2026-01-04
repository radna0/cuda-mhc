#!/bin/bash
# Launch Benchmark v36 (Safe Mode)
export HF_HUB_ENABLE_HF_TRANSFER=1
modal run --detach modal/mhc_training_benchmark.py::main
