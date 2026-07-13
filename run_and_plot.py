"""
Run one or more metrics on one or more synthetic tests and save the results
(a scores CSV and a static plot per test/metric pair) to a directory.

Tests and metrics are looked up by name in pyst.registry / metrics.registry,
which auto-discover every test/metric class in the pyst/ and metrics/
packages - no manual registration needed.

Each test's alignment score is computed automatically (via the test's own
compute_alignment_score(), see pyst/synthetic_test.py and the overrides in
pyst/ContrastMatching.py) and merged into alignment_scores/<test's preview
folder>.csv, shown in the plot title where available. There's no separate
scoring step to run.

Examples:
    python run_and_plot.py --list-tests
    python run_and_plot.py --list-metrics

    python run_and_plot.py \\
        --tests GaborPatchSpatialFreqAchTest ContrastMaskingCoherentAchTest \\
        --metrics cvvdp_met psnry_metric ssim_metric \\
        --output-dir results/my_run

    python run_and_plot.py --config my_run.json
        # my_run.json: {"tests": [...], "metrics": [...], "output_dir": "...", "alignment_score": false}
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


def run_and_plot(tests, metrics, output_dir, alignment_score=False):
    """
    tests: list of test classes (from pyst.registry)
    metrics: list of metric classes (from metrics.registry)
    output_dir: directory to save scores/plots into (created if missing)
    alignment_score: run tests in alignment-score mode (not supported by every test)

    Writes <output_dir>/<test.get_preview_folder()>/<metric_name>.csv (condition
    columns + a 'Prediction' column) and, where the test supports it, a matching
    <metric_name>.png plot in the same folder, plus an updated
    alignment_scores/<test.get_preview_folder()>.csv. This is the same layout the
    old per-test scripts produced, so results here are readable by
    generate_webpage.py.
    """
    os.makedirs(output_dir, exist_ok=True)

    metric_instances = [M() for M in metrics]
    metric_names = [M.name() for M in metrics]

    for T in tests:
        try:
            Test = T(is_alignmenet_score=True) if alignment_score else T()
        except TypeError as e:
            logging.error(f"Skipping {T.__name__}: doesn't support alignment_score={alignment_score} ({e})")
            continue

        test_name = Test.short_name()
        logging.info(f"Running '{test_name}' ({len(Test)} conditions) with {len(metrics)} metric(s)")

        test_output_dir = os.path.join(output_dir, Test.get_preview_folder())
        os.makedirs(test_output_dir, exist_ok=True)

        scores = np.full((len(Test), len(metrics)), np.nan)

        for i in trange(len(Test), desc=test_name):
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
                    logging.exception(f"Failed on '{test_name}', condition {i}, metric {metric_names[j]}")

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
            csv_path = os.path.join(test_output_dir, f"{metric_names[j]}.csv")
            results_df.to_csv(csv_path)
            logging.info(f"Saved scores: {csv_path}")

            align_score = None
            try:
                align_score = Test.compute_alignment_score(scores[:, j])
            except Exception:
                logging.exception(f"Failed to compute alignment score for '{test_name}' / {metric_names[j]}")
            if align_score is not None:
                alignment_scores[metric_names[j]] = align_score

            if can_plot:
                title = f"{metric_names[j]} — {test_name}"
                if align_score is not None:
                    title += f"\nAlignment Score = {align_score:.3f}"

                plot_path = os.path.join(test_output_dir, f"{metric_names[j]}.png")
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
            _update_alignment_scores_csv(Test.get_preview_folder(), alignment_scores)


def _load_config(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tests', nargs='+', metavar='TEST', help="Test class name(s) to run. See --list-tests.")
    parser.add_argument('--metrics', nargs='+', metavar='METRIC', help="Metric name(s) to run. See --list-metrics.")
    parser.add_argument('--output-dir', help="Directory to save plots/scores into (created if it doesn't exist).")
    parser.add_argument('--config', help="JSON file with 'tests', 'metrics', 'output_dir' and optional "
                                          "'alignment_score' keys, as an alternative to the flags above.")
    parser.add_argument('--alignment-score', action='store_true',
                         help="Run tests in alignment-score mode. Not supported by every test "
                              "(e.g. ContrastConstancySpatialFreqAchTest, ContrastConstancyColorTest) - "
                              "unsupported tests are skipped with an error message.")
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
        output_dir = cfg['output_dir']
        alignment_score = cfg.get('alignment_score', False)
    else:
        if not args.tests or not args.metrics or not args.output_dir:
            parser.error("--tests, --metrics and --output-dir are required unless --config is given.")
        test_names = args.tests
        metric_names = args.metrics
        output_dir = args.output_dir
        alignment_score = args.alignment_score

    try:
        test_classes = [tests_registry[name] for name in test_names]
    except KeyError as e:
        parser.error(f"Unknown test {e}. Use --list-tests to see available tests.")

    try:
        metric_classes = [metrics_registry[name] for name in metric_names]
    except KeyError as e:
        parser.error(f"Unknown metric {e}. Use --list-metrics to see available metrics.")

    run_and_plot(test_classes, metric_classes, output_dir, alignment_score=alignment_score)


if __name__ == '__main__':
    main()
