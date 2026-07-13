"""
Run one or more metrics on one or more synthetic tests and save the results
(a scores CSV per test/metric pair, plus a static plot) to disk.

Tests and metrics are looked up by name in pyst.registry / metrics.registry,
which auto-discover every test/metric class in the pyst/ and metrics/
packages - no manual registration needed.

Every test is run in its normal (full condition range) mode, producing the
main results/plots. If the test also supports alignment-score mode (a
near-threshold condition sampling), it's additionally run that way
automatically - no flag needed. Tests that don't support it (e.g. the
Contrast Matching tests) are simply only run once; their normal-mode run is
what produces their alignment score instead. Either way, each test's
alignment score is computed automatically (via the test's own
compute_alignment_score(), see pyst/synthetic_test.py and the overrides in
pyst/ContrastMatching.py) and merged into alignment_scores/<test's preview
folder>.csv - there's no separate scoring step to run.

Scores CSVs always go to a fixed location - results/<test's preview folder>
for the normal-mode run, results_alignment_score/... for the alignment-score
one - the same location the old per-test scripts used, and what
generate_webpage.py reads from by default. This isn't configurable. Only the
*plots* go to the directory you choose with --plots-dir.

Examples:
    python run_and_plot.py --list-tests
    python run_and_plot.py --list-metrics

    python run_and_plot.py \\
        --tests GaborPatchSpatialFreqAchTest ContrastMaskingCoherentAchTest \\
        --metrics cvvdp_met psnry_metric ssim_metric \\
        --plots-dir plots/my_run

    python run_and_plot.py --config my_run.json
        # my_run.json: {"tests": [...], "metrics": [...], "plots_dir": "..."}
"""
import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from tqdm import trange

from metrics.registry import get_metrics_registry
from pyst.registry import get_tests_registry

RESULTS_DIR = 'results'
ALIGNMENT_SCORE_RESULTS_DIR = 'results_alignment_score'
ALIGNMENT_SCORES_DIR = 'alignment_scores'


def _update_alignment_scores_csv(preview_folder, metric_scores):
    """
    Merge metric_scores ({metric_name: score}) into
    alignment_scores/<preview_folder>.csv, keeping any existing rows for
    metrics not in this run (so results from separate runs accumulate,
    rather than each run wiping out the others').
    """
    csv_path = os.path.join(ALIGNMENT_SCORES_DIR, preview_folder + '.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path).set_index('Metric')['Score'].to_dict()
    else:
        existing = {}

    existing.update(metric_scores)

    df = pd.DataFrame({'Metric': list(existing.keys()), 'Score': list(existing.values())})
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved alignment scores: {csv_path}")


def _run_one_mode(T, metric_instances, metric_names, plots_dir, alignment_score):
    """
    Instantiate and run test class T, in alignment-score mode if requested.
    Returns silently (does nothing) if T doesn't support that mode.
    """
    try:
        Test = T(is_alignmenet_score=True) if alignment_score else T()
    except TypeError as e:
        if alignment_score:
            logging.info(f"{T.__name__} doesn't support alignment-score mode, skipping that pass ({e})")
        else:
            logging.error(f"Skipping {T.__name__}: couldn't even instantiate it in normal mode ({e})")
        return

    test_name = Test.short_name()
    mode_label = "alignment-score" if alignment_score else "normal"
    logging.info(f"Running '{test_name}' in {mode_label} mode ({len(Test)} conditions) with {len(metric_names)} metric(s)")

    preview_folder = Test.get_preview_folder()

    results_dir = ALIGNMENT_SCORE_RESULTS_DIR if alignment_score else RESULTS_DIR
    results_test_dir = os.path.join(results_dir, preview_folder)
    os.makedirs(results_test_dir, exist_ok=True)

    plots_test_dir = os.path.join(plots_dir, preview_folder)
    os.makedirs(plots_test_dir, exist_ok=True)

    scores = np.full((len(Test), len(metric_names)), np.nan)

    for i in trange(len(Test), desc=f"{test_name} ({mode_label})"):
        vid_source, display_photometry, display_geometry = Test.get_condition(i)

        for j, Metric in enumerate(metric_instances):
            try:
                Metric.set_display_model(display_photometry=display_photometry, display_geometry=display_geometry)

                if hasattr(Test, 'run_on_test_condition'):
                    score = Test.run_on_test_condition(i, Metric)
                else:
                    with torch.no_grad():
                        score = Metric.predict_video_source(vid_source)

                if isinstance(score, tuple) and len(score) == 2:
                    score, _ = score

                scores[i, j] = score
            except Exception:
                logging.exception(f"Failed on '{test_name}' ({mode_label}), condition {i}, metric {metric_names[j]}")

    header = Test.get_row_header()
    conditions = Test.get_rows_conditions()

    results_df = pd.DataFrame()
    results_df.index.name = 'Index'
    for i in range(len(header)):
        try:
            results_df[header[i]] = conditions[i].cpu().numpy()
        except AttributeError:
            results_df[header[i]] = conditions[i]

    can_plot = hasattr(Test, 'plot')
    if not can_plot:
        logging.warning(
            f"{type(Test).__name__} has no plot() method yet, so static plots can't be generated for it "
            f"(scores are still saved to CSV). Use generate_webpage.py for an "
            f"interactive plotly version instead."
        )

    alignment_scores = {}

    for j, Metric in enumerate(metric_instances):
        results_df['Prediction'] = scores[:, j]
        csv_path = os.path.join(results_test_dir, f"{metric_names[j]}.csv")
        results_df.to_csv(csv_path)
        logging.info(f"Saved scores: {csv_path}")

        align_score = None
        try:
            align_score = Test.compute_alignment_score(scores[:, j])
        except Exception:
            logging.exception(f"Failed to compute alignment score for '{test_name}' ({mode_label}) / {metric_names[j]}")
        if align_score is not None:
            alignment_scores[metric_names[j]] = align_score

        if can_plot:
            title = f"{metric_names[j]} — {test_name} ({mode_label})"
            if align_score is not None:
                title += f"\nAlignment Score = {align_score:.3f}"

            plot_path = os.path.join(plots_test_dir, f"{metric_names[j]}_{mode_label}.png")
            try:
                Test.plot(
                    predictions=scores[:, j],
                    reverse_color_order=Metric.is_lower_better(),
                    title=title,
                    output_filename=plot_path,
                )
                logging.info(f"Saved plot: {plot_path}")
            except Exception:
                logging.exception(f"Failed to plot '{test_name}' ({mode_label}) / {metric_names[j]}")

    if alignment_scores:
        _update_alignment_scores_csv(preview_folder, alignment_scores)


def run_and_plot(tests, metrics, plots_dir):
    """
    tests: list of test classes (from pyst.registry)
    metrics: list of metric classes (from metrics.registry)
    plots_dir: directory to save plots into (created if missing) - scores
        CSVs do NOT go here, see module docstring

    Each test is run in normal mode, and additionally in alignment-score mode
    if it supports it - see module docstring for exactly what that writes
    where.
    """
    metric_instances = [M() for M in metrics]
    metric_names = [M.name() for M in metrics]

    for T in tests:
        _run_one_mode(T, metric_instances, metric_names, plots_dir, alignment_score=False)
        _run_one_mode(T, metric_instances, metric_names, plots_dir, alignment_score=True)


def _load_config(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tests', nargs='+', metavar='TEST', help="Test class name(s) to run. See --list-tests.")
    parser.add_argument('--metrics', nargs='+', metavar='METRIC', help="Metric name(s) to run. See --list-metrics.")
    parser.add_argument('--plots-dir', help="Directory to save plots into (created if it doesn't exist). "
                                             "Scores CSVs always go to a fixed results/ (or results_alignment_score/) "
                                             "location instead - see the module docstring.")
    parser.add_argument('--config', help="JSON file with 'tests', 'metrics' and 'plots_dir' keys, as an "
                                          "alternative to the flags above.")
    parser.add_argument('--list-tests', action='store_true', help="Print all available test names and exit.")
    parser.add_argument('--list-metrics', action='store_true', help="Print all available metric names and exit.")
    args = parser.parse_args()

    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

    tests_registry = get_tests_registry()
    metrics_registry = get_metrics_registry()

    if args.list_tests:
        print('\n'.join(sorted(tests_registry)))
        return

    if args.list_metrics:
        print('\n'.join(sorted(metrics_registry)))
        return

    if args.config:
        cfg = _load_config(args.config)
        test_names = cfg['tests']
        metric_names = cfg['metrics']
        plots_dir = cfg['plots_dir']
    else:
        if not args.tests or not args.metrics or not args.plots_dir:
            parser.error("--tests, --metrics and --plots-dir are required unless --config is given.")
        test_names = args.tests
        metric_names = args.metrics
        plots_dir = args.plots_dir

    try:
        test_classes = [tests_registry[name] for name in test_names]
    except KeyError as e:
        parser.error(f"Unknown test {e}. Use --list-tests to see available tests.")

    try:
        metric_classes = [metrics_registry[name] for name in metric_names]
    except KeyError as e:
        parser.error(f"Unknown metric {e}. Use --list-metrics to see available metrics.")

    run_and_plot(test_classes, metric_classes, plots_dir)


if __name__ == '__main__':
    main()
