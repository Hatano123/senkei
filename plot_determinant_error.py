import csv
import math
import numpy as np
import matplotlib.pyplot as plt


def main():
    csv_path = "experiment_results.csv"
    out_png = "determinant_error.png"

    sizes = []
    errors = []

    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sizes.append(int(row["Matrix Size"]))
                errors.append(float(row["Determinant Error"]))
            except Exception:
                # skip malformed rows
                continue

    if not sizes:
        raise SystemExit("CSV に有効なデータが見つかりませんでした。")

    x = np.array(sizes)
    y = np.array(errors)

    # Clean data
    finite_mask = np.isfinite(y)
    n_nonfinite = np.count_nonzero(~finite_mask)
    if n_nonfinite:
        print(f"Warning: {n_nonfinite} non-finite error values found and will be skipped.")

    x = x[finite_mask]
    y = y[finite_mask]

    # Identify positive values for log plotting
    positive_mask = y > 0
    y_pos = y.copy()
    y_pos[~positive_mask] = np.nan

    # If all positive values are extreme, clip at a reasonable percentile for visualization
    if np.any(~np.isnan(y_pos)):
        log10_vals = np.log10(y_pos[~np.isnan(y_pos)])
        p95 = np.nanpercentile(log10_vals, 95)
        clip_exp = p95 + 1.0  # allow a margin above 95th percentile
        # cap exponent to avoid overflow when computing 10**exp
        clip_exp = float(np.minimum(clip_exp, 300.0))
        clip_val = float(10.0 ** clip_exp)
        n_clipped = np.count_nonzero(y_pos > clip_val)
    else:
        clip_val = None
        n_clipped = 0

    # Main figure: log-scale with clipping (if necessary)
    plt.figure(figsize=(9, 5))
    # For log plot show only positive values and clip extreme ones to avoid overflow
    if clip_val is not None:
        y_plot = np.where(y > 0, np.minimum(y, clip_val), np.nan)
        label_text = f'Positive (clipped at {clip_val:.2e})'
    else:
        y_plot = np.where(y > 0, y, np.nan)
        label_text = 'Positive values'

    # To avoid log-scale overflow in matplotlib, plot log10(error) on a linear axis
    log10_plot = np.where(np.isfinite(y_plot) & (y_plot > 0), np.log10(y_plot), np.nan)

    plt.scatter(x, log10_plot, c='C1', s=18, label=label_text)

    if clip_val is not None:
        # determine reasonable plot limits in log10 (linear) space
        min_exp = float(np.nanmin(log10_vals)) if np.any(~np.isnan(y_pos)) else -300.0
        bottom_exp = max(min_exp - 1.0, -300.0)
        top_exp = min(clip_exp + 1.0, 300.0)
        plt.ylim(bottom_exp, top_exp)

    plt.xlabel('Matrix Size')
    plt.ylabel('log10(Determinant Error)')
    plt.title('Determinant Error vs Matrix Size')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    try:
        plt.tight_layout()
    except Exception:
        pass
    out1 = 'determinant_error_clipped.png'
    plt.savefig(out1, dpi=150)
    print(f"Saved plot to: {out1}")

    # Zoomed plot: remove top outliers to show trend
    if clip_val is not None:
        mask_zoom = (y_pos <= clip_val)
        if np.count_nonzero(mask_zoom) > 2:
            plt.figure(figsize=(9, 5))
            # plot log10 of zoomed positive values on linear axis
            y_zoom = y_pos[mask_zoom]
            log10_zoom = np.where(np.isfinite(y_zoom) & (y_zoom > 0), np.log10(y_zoom), np.nan)
            plt.plot(x[mask_zoom], log10_zoom, marker='.', linestyle='-', alpha=0.7, color='red', markeredgecolor='red')
            plt.xlabel('Matrix Size')
            plt.ylabel('log10(Determinant Error)')
            plt.title('Determinant Error (zoom, outliers removed)')
            plt.grid(True, which='both', ls='--', lw=0.5)
            try:
                plt.tight_layout()
            except Exception:
                pass
            out2 = 'determinant_error_zoom.png'
            plt.savefig(out2, dpi=150)
            print(f"Saved zoomed plot to: {out2}")

    # Histogram of log10 errors (positive values only)
    if np.any(~np.isnan(y_pos)):
        plt.figure(figsize=(7, 4))
        vals = np.log10(y_pos[~np.isnan(y_pos)])
        plt.hist(vals, bins=60, color='C2', alpha=0.8)
        plt.xlabel('log10(Determinant Error)')
        plt.ylabel('Count')
        plt.title('Histogram of log10(Determinant Error)')
        try:
            plt.tight_layout()
        except Exception:
            pass
        out3 = 'determinant_error_hist.png'
        plt.savefig(out3, dpi=150)
        print(f"Saved histogram to: {out3}")

    # Summary
    total = len(errors)
    print(f"Total rows: {total}, non-finite skipped: {n_nonfinite}, clipped: {n_clipped}")


if __name__ == '__main__':
    main()
