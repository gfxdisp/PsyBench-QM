"""Plot ColorVideoVDP vs other metrics for gabor ambient-light results.

Usage:
    python scripts/plot_ambient_scatter.py [--show] [--out-dir <path>]

Produces one PNG per metric found in results/gabor_tests/ambient_light
(excluding ColorVideoVDP.csv itself). Points are colored by ambient
illumination (log color scale). Each metric is saved to a separate file
in the output directory.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_scatter(xvals, yvals, ambient_vals, title, out_file=None, show=False):
    fig, ax = plt.subplots(figsize=(6, 5))

    vmin = max(np.nanmin(ambient_vals), 1e-3)
    vmax = np.nanmax(ambient_vals)
    sc = ax.scatter(xvals, yvals, c=ambient_vals, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax), edgecolors='none')
    ax.set_xlabel('ColorVideoVDP Prediction')
    ax.set_ylabel(title + ' Prediction')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Fit a linear regression line for reference if enough points
    try:
        if len(xvals) >= 2:
            coeffs = np.polyfit(xvals, yvals, 1)
            xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 100)
            ax.plot(xs, coeffs[0] * xs + coeffs[1], color='red', linewidth=1)
    except Exception:
        pass

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Ambient Illumination (lux)')

    fig.tight_layout()
    if out_file:
        fig.savefig(out_file, dpi=200)
        print(f'Saved {out_file}')

    if show:
        plt.show()

    plt.close(fig)


def main(data_dir='results/gabor_tests/ambient_light', out_dir=None, show=False):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    if 'ColorVideoVDP.csv' not in files:
        raise FileNotFoundError(f"Base file ColorVideoVDP.csv not found in {data_dir}")

    base = pd.read_csv(os.path.join(data_dir, 'ColorVideoVDP.csv'))
    # Prefer known ambient column names used earlier
    ambient_col_candidates = ['Ambient Illumination', 'Ambient', 'E_ambient', 'AmbientIlluminance']
    ambient_col = next((c for c in ambient_col_candidates if c in base.columns), None)
    if ambient_col is None:
        raise KeyError('Ambient Illumination column not found in ColorVideoVDP.csv')

    ambient = base[ambient_col].astype(float)
    x = base['Prediction'].astype(float)

    other_files = [f for f in files if f != 'ColorVideoVDP.csv']
    if len(other_files) == 0:
        print('No other metric files found to compare.')
        return

    if out_dir is None:
        out_dir = os.path.join(data_dir, 'scatter_plots')
    os.makedirs(out_dir, exist_ok=True)

    for fname in other_files:
        df = pd.read_csv(os.path.join(data_dir, fname))

        # Align rows by Index if present in both; otherwise rely on row order
        if 'Index' in df.columns and 'Index' in base.columns:
            merged = base[['Index', 'Prediction', ambient_col]].merge(
                df[['Index', 'Prediction']], on='Index', suffixes=('_ColorVideoVDP', '_other'))
            xvals = merged['Prediction_ColorVideoVDP'].astype(float).to_numpy()
            yvals = merged['Prediction_other'].astype(float).to_numpy()
            ambient_vals = merged[ambient_col].astype(float).to_numpy()
        else:
            xvals = x.to_numpy()
            yvals = df['Prediction'].astype(float).to_numpy()
            ambient_vals = ambient.to_numpy()

        title = fname.replace('.csv', '')
        safe_name = title.replace(' ', '_')
        out_file = os.path.join(out_dir, f'scatter_ColorVideoVDP_vs_{safe_name}.png')

        plot_scatter(xvals, yvals, ambient_vals, title, out_file=out_file, show=show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='results/gabor_tests/ambient_light')
    parser.add_argument('--out-dir', default=None, help='Directory to save per-metric plots (default: data_dir/scatter_plots)')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir, show=args.show)
"""Plot ColorVideoVDP vs other metrics for gabor ambient-light results.

Usage:
    python scripts/plot_ambient_scatter.py [--show] [--out <path>]

Produces one subplot per metric found in results/gabor_tests/ambient_light
(excluding ColorVideoVDP.csv itself). Points are colored by ambient
illumination (log color scale).
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main(data_dir='results/gabor_tests/ambient_light', out_pat=None, show=False):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    if 'ColorVideoVDP.csv' not in files:
        raise FileNotFoundError(f"Base file ColorVideoVDP.csv not found in {data_dir}")

    base = pd.read_csv(os.path.join(data_dir, 'ColorVideoVDP.csv'))
    # Ensure Ambient Illumination column exists and is numeric
    if 'Ambient Illumination' not in base.columns:
        raise KeyError('Ambient Illumination column not found in ColorVideoVDP.csv')
    ambient = base['Ambient Illumination'].astype(float)
    x = base['Prediction'].astype(float)

    other_files = [f for f in files if f != 'ColorVideoVDP.csv']
    n = len(other_files)
    if n == 0:
        print('No other metric files found to compare.')
        return

    # Prepare subplot grid
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    vmin = ambient.min()
    vmax = ambient.max()

    for idx, fname in enumerate(other_files):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        df = pd.read_csv(os.path.join(data_dir, fname))
        # Align using Index if present, else rely on order
        if 'Index' in df.columns and 'Index' in base.columns:
            merged = base[['Index', 'Prediction']].merge(df[['Index', 'Prediction']], on='Index', suffixes=('_ColorVideoVDP', '_other'))
            xvals = merged['Prediction_ColorVideoVDP'].astype(float)
            yvals = merged['Prediction_other'].astype(float)
            ambient_vals = merged['Index'].map(lambda i: base.loc[base['Index'] == i, 'Ambient Illumination'].values[0]).astype(float) if 'Index' in base.columns else ambient
        else:
            xvals = x
            """Plot ColorVideoVDP vs other metrics for gabor ambient-light results.

            Usage:
                python scripts/plot_ambient_scatter.py [--show] [--out-dir <path>]

            Produces one PNG per metric found in results/gabor_tests/ambient_light
            (excluding ColorVideoVDP.csv itself). Points are colored by ambient
            illumination (log color scale). Each metric is saved to a separate file
            in the output directory.
            """
            import os
            import argparse
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm


            def plot_scatter(xvals, yvals, ambient_vals, title, out_file=None, show=False):
                fig, ax = plt.subplots(figsize=(6, 5))

                vmin = max(np.nanmin(ambient_vals), 1e-3)
                vmax = np.nanmax(ambient_vals)
                sc = ax.scatter(xvals, yvals, c=ambient_vals, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax), edgecolors='none')
                ax.set_xlabel('ColorVideoVDP Prediction')
                ax.set_ylabel(title + ' Prediction')
                ax.set_title(title)
                ax.grid(True, linestyle='--', alpha=0.3)

                # Fit a linear regression line for reference if enough points
                try:
                    if len(xvals) >= 2:
                        coeffs = np.polyfit(xvals, yvals, 1)
                        xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 100)
                        ax.plot(xs, coeffs[0] * xs + coeffs[1], color='red', linewidth=1)
                except Exception:
                    pass

                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label('Ambient Illumination (lux)')

                fig.tight_layout()
                if out_file:
                    fig.savefig(out_file, dpi=200)
                    print(f'Saved {out_file}')

                if show:
                    plt.show()

                plt.close(fig)


            def main(data_dir='results/gabor_tests/ambient_light', out_dir=None, show=False):
                files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
                if 'ColorVideoVDP.csv' not in files:
                    raise FileNotFoundError(f"Base file ColorVideoVDP.csv not found in {data_dir}")

                base = pd.read_csv(os.path.join(data_dir, 'ColorVideoVDP.csv'))
                # Prefer known ambient column names used earlier
                ambient_col_candidates = ['Ambient Illumination', 'Ambient', 'E_ambient', 'AmbientIlluminance']
                ambient_col = next((c for c in ambient_col_candidates if c in base.columns), None)
                if ambient_col is None:
                    raise KeyError('Ambient Illumination column not found in ColorVideoVDP.csv')

                ambient = base[ambient_col].astype(float)
                x = base['Prediction'].astype(float)

                other_files = [f for f in files if f != 'ColorVideoVDP.csv']
                if len(other_files) == 0:
                    print('No other metric files found to compare.')
                    return

                if out_dir is None:
                    out_dir = os.path.join('scatter_plots')
                os.makedirs(out_dir, exist_ok=True)

                for fname in other_files:
                    df = pd.read_csv(os.path.join(data_dir, fname))

                    # Align rows by Index if present in both; otherwise rely on row order
                    if 'Index' in df.columns and 'Index' in base.columns:
                        merged = base[['Index', 'Prediction', ambient_col]].merge(
                            df[['Index', 'Prediction']], on='Index', suffixes=('_ColorVideoVDP', '_other'))
                        xvals = merged['Prediction_ColorVideoVDP'].astype(float).to_numpy()
                        yvals = merged['Prediction_other'].astype(float).to_numpy()
                        ambient_vals = merged[ambient_col].astype(float).to_numpy()
                    else:
                        xvals = x.to_numpy()
                        yvals = df['Prediction'].astype(float).to_numpy()
                        ambient_vals = ambient.to_numpy()

                    title = fname.replace('.csv', '')
                    safe_name = title.replace(' ', '_')
                    out_file = os.path.join(out_dir, f'scatter_ColorVideoVDP_vs_{safe_name}.png')

                    print(f'Plotting ColorVideoVDP vs {title}... in {out_file}')
                    plot_scatter(xvals, yvals, ambient_vals, title, out_file=out_file, show=show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='results/gabor_tests/ambient_light')
    parser.add_argument('--out-dir', default=None, help='Directory to save per-metric plots (default: data_dir/scatter_plots)')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir, show=args.show)
