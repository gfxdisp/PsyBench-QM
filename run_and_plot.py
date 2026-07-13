"""
Run one or more metrics on one or more synthetic tests and save the results
(a scores CSV per test/metric pair, plus a static plot) to disk.

Tests and metrics are looked up by name in pyst.registry / metrics.registry,
which auto-discover every test/metric class in the pyst/ and metrics/
packages - no manual registration needed.

Every test is run in its normal (full condition range) mode, producing the
main results and a plot. If the test also supports alignment-score mode (a
near-threshold condition sampling), it's additionally run that way
automatically - no flag needed - but that pass exists purely to measure
alignment; it's not plotted. Tests that don't support alignment-score mode
(e.g. the Contrast Matching tests) are simply only run once; their
normal-mode run is what produces their alignment score instead. Either way,
each test's alignment score is computed automatically (via the test's own
compute_alignment_score(), see pyst/synthetic_test.py and the overrides in
pyst/ContrastMatching.py) and merged into alignment_scores/<test's preview
folder>.csv - there's no separate scoring step to run.

Scores CSVs always go to a fixed location - results/<test's preview folder>
for the normal-mode run, results_alignment_score/... for the alignment-score
one - the same location the old per-test scripts used, and what
generate_webpage.py reads from by default. This isn't configurable. Only the
*plot* (normal-mode only) goes to the directory you choose with --plots-dir.

Examples:
    python run_and_plot.py --list-tests
    python run_and_plot.py --list-metrics

    python run_and_plot.py \\
        --tests GaborPatchSpatialFreqAchTest ContrastMaskingCoherentAchTest \\
        --metrics ColorVideoVDP PSNR-Y SSIM \\
        --plots-dir plots/my_run

    python run_and_plot.py --config my_run.json
        # my_run.json: {"tests": [...], "metrics": [...], "plots_dir": "...", "plot_only": false}

    python run_and_plot.py --config my_run.json --plot-only
        # reuses existing results/ (and results_alignment_score/) instead of re-running metrics
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


def _run_one_mode(T, metric_instances, metric_names, plots_dir, alignment_score, plot_only=False):
    """
    Instantiate and run test class T, in alignment-score mode if requested.
    Returns silently (does nothing) if T doesn't support that mode.

    If plot_only, metrics aren't actually run - predictions are instead read
    back from the existing results CSVs (results/... or
    results_alignment_score/..., matching the mode), so this only works if
    run_and_plot.py already wrote them there in a previous, non-plot-only run.
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

    preview_folder = Test.get_preview_folder()

    results_dir = ALIGNMENT_SCORE_RESULTS_DIR if alignment_score else RESULTS_DIR
    results_test_dir = os.path.join(results_dir, preview_folder)

    if not alignment_score:
        plots_test_dir = os.path.join(plots_dir, preview_folder)
        os.makedirs(plots_test_dir, exist_ok=True)

    scores = np.full((len(Test), len(metric_names)), np.nan)

    if plot_only:
        if not os.path.isdir(results_test_dir):
            logging.error(f"--plot-only: no existing results at {results_test_dir} for '{test_name}' "
                           f"({mode_label} mode) - run without --plot-only first. Skipping.")
            return

        logging.info(f"Reading existing results for '{test_name}' ({mode_label} mode) instead of running metrics")
        for j, name in enumerate(metric_names):
            csv_path = os.path.join(results_test_dir, f"{name}.csv")
            if not os.path.exists(csv_path):
                logging.error(f"--plot-only: missing {csv_path}, skipping metric {name} for '{test_name}' ({mode_label})")
                continue
            scores[:, j] = pd.read_csv(csv_path)['Prediction'].to_numpy()

    else:
        logging.info(f"Running '{test_name}' in {mode_label} mode ({len(Test)} conditions) with {len(metric_names)} metric(s)")
        os.makedirs(results_test_dir, exist_ok=True)

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

        for j in range(len(metric_names)):
            results_df['Prediction'] = scores[:, j]
            csv_path = os.path.join(results_test_dir, f"{metric_names[j]}.csv")
            results_df.to_csv(csv_path)
            logging.info(f"Saved scores: {csv_path}")

    can_plot = hasattr(Test, 'plot') and not alignment_score
    if not can_plot and not alignment_score:
        logging.warning(
            f"{type(Test).__name__} has no plot() method yet, so static plots can't be generated for it "
            f"(scores are still saved to CSV). Use generate_webpage.py for an "
            f"interactive plotly version instead."
        )

    alignment_scores = {}

    for j, Metric in enumerate(metric_instances):
        align_score = None
        try:
            align_score = Test.compute_alignment_score(scores[:, j])
        except Exception:
            logging.exception(f"Failed to compute alignment score for '{test_name}' ({mode_label}) / {metric_names[j]}")
        if align_score is not None:
            alignment_scores[metric_names[j]] = align_score

        if can_plot:
            title = f"{metric_names[j]} — {test_name}"
            if align_score is not None:
                title += f"\nAlignment Score = {align_score:.3f}"

            plot_path = os.path.join(plots_test_dir, f"{metric_names[j]}.png")
            try:
                Test.plot(
                    predictions=scores[:, j],
                    reverse_color_order=Metric.is_lower_better(),
                    title=title,
                    output_filename=plot_path,
                )
                logging.info(f"Saved plot: {plot_path}")
            except Exception:
                logging.exception(f"Failed to plot '{test_name}' / {metric_names[j]}")

    if alignment_scores:
        _update_alignment_scores_csv(preview_folder, alignment_scores)


def run_and_plot(tests, metrics, plots_dir, plot_only=False):
    """
    tests: list of test classes (from pyst.registry)
    metrics: list of metric classes (from metrics.registry)
    plots_dir: directory to save plots into (created if missing) - scores
        CSVs do NOT go here, see module docstring
    plot_only: don't run any metrics - just (re)generate plots and alignment
        scores from whatever's already in results/ / results_alignment_score/
        from a previous run. Fails per-test/mode if that data isn't there.

    Each test is run in normal mode, and additionally in alignment-score mode
    if it supports it - see module docstring for exactly what that writes
    where.
    """
    metric_instances = [M() for M in metrics]
    metric_names = [M.name() for M in metrics]

    for T in tests:
        _run_one_mode(T, metric_instances, metric_names, plots_dir, alignment_score=False, plot_only=plot_only)
        _run_one_mode(T, metric_instances, metric_names, plots_dir, alignment_score=True, plot_only=plot_only)


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
    parser.add_argument('--config', help="JSON file with 'tests', 'metrics', 'plots_dir' and optional "
                                          "'plot_only' keys, as an alternative to the flags above.")
    parser.add_argument('--plot-only', action='store_true',
                         help="Don't run any metrics - just (re)generate plots and alignment scores from "
                              "results already on disk (results/ and results_alignment_score/, from a previous "
                              "non---plot-only run). Useful if you already have results and just want new plots, "
                              "e.g. after a plot()-generation code change.")
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
        plot_only = cfg.get('plot_only', False)
    else:
        if not args.tests or not args.metrics or not args.plots_dir:
            parser.error("--tests, --metrics and --plots-dir are required unless --config is given.")
        test_names = args.tests
        metric_names = args.metrics
        plots_dir = args.plots_dir
        plot_only = args.plot_only

    try:
        test_classes = [tests_registry[name] for name in test_names]
    except KeyError as e:
        parser.error(f"Unknown test {e}. Use --list-tests to see available tests.")

    try:
        metric_classes = [metrics_registry[name] for name in metric_names]
    except KeyError as e:
        parser.error(f"Unknown metric {e}. Use --list-metrics to see available metrics.")

    run_and_plot(test_classes, metric_classes, plots_dir, plot_only=plot_only)


if __name__ == '__main__':
    main()
