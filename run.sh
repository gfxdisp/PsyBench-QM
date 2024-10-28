#!/bin/bash

python run_synthetic_test.py
python generate_synthetic_results_webpage.py
./update_webpage_public.sh