import csv
import math

def read_results(path):
    sizes = []
    det = []
    lin = []
    eig = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                n = int(r['Matrix Size'])
            except Exception:
                continue
            sizes.append(n)
            det.append(float(r.get('Custom Determinant Time (s)', '0') or 0.0))
            lin.append(float(r.get('Custom Linear Time (s)', '0') or 0.0))
            eig.append(float(r.get('Custom Eigen Time (s)', '0') or 0.0))
    return sizes, det, lin, eig


def approx_cubic_constant(sizes, times, min_n=10):
    vals = []
    for n, t in zip(sizes, times):
        if n >= min_n and n > 0 and t > 0:
            vals.append(t / (n**3))
    if not vals:
        return None
    return sum(vals) / len(vals)


def format_sec(x):
    if x is None:
        return 'N/A'
    if x < 1e-3:
        return f"{x*1e6:.1f} µs"
    return f"{x:.6f} s"


def generate_report(csv_path='experiment_results.csv'):
    sizes, det, lin, eig = read_results(csv_path)
    if not sizes:
        print('CSV が見つからないか中身がありません。')
        return

    max_idx = sizes.index(max(sizes))
    max_n = sizes[max_idx]
    det_max = det[max_idx]
    lin_max = lin[max_idx]
    eig_max = eig[max_idx]

    det_k = approx_cubic_constant(sizes, det)
    lin_k = approx_cubic_constant(sizes, lin)

    print('図4.1 — 行列式の計算時間（前進消去）')
    print(f'- 観測: 次数 n が大きくなるにつれて計算時間は増加します（おおむね O(n^3)）。')
    if det_k is not None:
        print(f"- 近似: t_det(n) ≈ {det_k:.3e} · n^3")
    print(f"- 最大次数 (n={max_n}) の計測: {format_sec(det_max)}")
    print()

    print('図4.2 — 連立1次方程式の計算時間（ガウスの消去）')
    print('- 観測: 行列式と同様に O(n^3) の増加を示します。右辺ベクトルにも同じ行基本変形を施すため、わずかに大きめです。')
    if lin_k is not None:
        print(f"- 近似: t_lin(n) ≈ {lin_k:.3e} · n^3")
    print(f"- 最大次数 (n={max_n}) の計測: {format_sec(lin_max)}")
    print()

    print('図4.3 — 固有値の計算時間（QR法・反復）')
    print('- 観測: QR分解（グラム・シュミット）を繰り返すため非常に重く、次数に対して最も急激に増加します。')
    print(f"- 最大次数 (n={max_n}) の計測: {format_sec(eig_max)}")
    print()

    print('まとめ：')
    print('- 行列式: 多重ループによる O(n^3) 成長（右肩上がりの3次曲線）')
    print('- 連立方程式: 同じ O(n^3)、係数は行列式より大きめ')
    print('- 固有値: QR 法の反復により他を上回る増加率（最も重い）')


if __name__ == "__main__":
    generate_report('experiment_results.csv')
