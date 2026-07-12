#!/bin/bash

# PYTHONPATH=../metric_calibration/ColorVideoVDP:../metric_calibration python run_synthetic_test.py 0
# # python run_synthetic_test.py 1
# # python run_synthetic_test.py 2

# PYTHONPATH=../metric_calibration/ColorVideoVDP:../metric_calibration python run_synthetic_tests_alignment_score.py 0
# # python run_synthetic_tests_alignment_score.py 1
# # python run_synthetic_tests_alignment_score.py 2

# PYTHONPATH=../metric_calibration/ColorVideoVDP:../metric_calibration python get_alignment_scores.py

# PYTHONPATH=../metric_calibration/ColorVideoVDP:../metric_calibration python generate_synthetic_results_webpage_plotly.py
# ./update_webpage_public.sh

# python get_alignment_scores_latex_table.py

PYTHONPATH=../glare_model/metric_calibration:../glare_model/metric_calibration/ColorVideoVDP python run_synthetic_test.py 3
PYTHONPATH=../glare_model/metric_calibration:../glare_model/metric_calibration/ColorVideoVDP python run_synthetic_tests_alignment_score.py 3

python get_alignment_scores.py

PYTHONPATH=../glare_model/metric_calibration:../glare_model/metric_calibration/ColorVideoVDP python generate_synthetic_results_webpage_plotly.py
./update_webpage_public.sh