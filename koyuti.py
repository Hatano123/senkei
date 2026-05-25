import csv
import math
import time
from fractions import Fraction
from math import gcd


def qr_decomposition(matrix):
    """グラム・シュミットの直交化法によるQR分解"""
    n = len(matrix)
    Q = [[0.0] * n for _ in range(n)]
    R = [[0.0] * n for _ in range(n)]

    for j in range(n):
        # v = Aのj番目の列ベクトル
        v = [matrix[i][j] for i in range(n)]

        for i in range(j):
            # R[i][j] = Qのi番目の列ベクトルとvの内積
            dot_product = sum(Q[k][i] * v[k] for k in range(n))
            R[i][j] = dot_product
            # vからQ[i]の成分を引き算
            for k in range(n):
                v[k] -= R[i][j] * Q[k][i]

        # R[j][j] = 残ったベクトルvの長さ（ノルム）
        norm = math.sqrt(sum(x**2 for x in v))
        R[j][j] = norm

        # Q[j] = 正規化したベクトル
        if norm > 1e-12:
            for k in range(n):
                Q[k][j] = v[k] / norm
        else:
            for k in range(n):
                Q[k][j] = 0.0

    return Q, R


def matrix_multiply(A, B):
    """行列同士の掛け算"""
    n = len(A)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
    return result


def qr_method(matrix, iterations=100):
    """QR法による全固有値の算出"""
    n = len(matrix)
    A_k = [row[:] for row in matrix]

    # 反復計算（分解して逆順に掛け直すループ）
    for _ in range(iterations):
        Q, R = qr_decomposition(A_k)
        A_k = matrix_multiply(R, Q)

    # ループ終了後、対角成分に固有値が並ぶ
    eigenvalues = [A_k[i][i] for i in range(n)]
    return eigenvalues


def power_method(matrix, iterations=100):
    """べき乗法による最大固有値と対応ベクトルの近似"""
    size = len(matrix)
    vector = [1.0 for _ in range(size)]
    eigenvalue = 0.0

    for _ in range(iterations):
        next_vector = []
        for i in range(size):
            total = 0.0
            for j in range(size):
                total += matrix[i][j] * vector[j]
            next_vector.append(total)

        eigenvalue = next_vector[0]
        for value in next_vector:
            if abs(value) > abs(eigenvalue):
                eigenvalue = value

        if abs(eigenvalue) < 1e-12:
            break

        for i in range(size):
            vector[i] = next_vector[i] / eigenvalue

    return eigenvalue, vector


def to_integer_vector(vec, max_denominator=1000):
    """小数のベクトルを綺麗な整数比に変換する"""
    fracs = []
    for v in vec:
        if abs(v) < 1e-12:
            fracs.append(Fraction(0, 1))
        else:
            fracs.append(Fraction(v).limit_denominator(max_denominator))

    dens = [f.denominator for f in fracs]
    l = 1
    for d in dens:
        l = l * d // gcd(l, d)

    ints = [f.numerator * (l // f.denominator) for f in fracs]
    if all(x == 0 for x in ints):
        return [0] * len(vec)

    g = 0
    for x in ints:
        g = gcd(g, abs(x))
    if g > 1:
        ints = [x // g for x in ints]

    # 最初の非ゼロ要素が正になるように符号を調整
    for x in ints:
        if x < 0:
            ints = [-y for y in ints]
            break
        elif x > 0:
            break

    return ints


def write_result_to_csv(output_file, rows):
    """結果をCSVファイルに保存"""
    # Reduce output: omit full matrix and integer eigenvectors to keep CSV compact
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "Method",
            "Matrix Size",
            "Elapsed Time (s)",
            "Eigenvalues",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Method": row.get("Method"),
                    "Matrix Size": row.get("Matrix Size"),
                    "Elapsed Time (s)": row.get("Elapsed Time (s)"),
                    "Eigenvalues": row.get("Eigenvalues"),
                }
            )


def main():
    matrix_size = int(input("行列の次数Nを入力: "))
    matrix = [list(map(float, input().split())) for _ in range(matrix_size)]

    # --- 1. 自作QR法による計算 ---
    qr_start_time = time.perf_counter()
    custom_eigenvalues = qr_method(matrix)
    qr_elapsed_time = time.perf_counter() - qr_start_time

    print("\n--- 自作QR法による固有値算出結果 ---")
    rounded_custom_eigenvalues = [round(val, 5) for val in custom_eigenvalues]
    print(f"算出された固有値一覧: {rounded_custom_eigenvalues}")
    print(f"自作QR法の計算時間: {qr_elapsed_time:.6f} 秒")

    # NumPyを使わず、カスタムQR法のみで結果を表示
    print("\n--- 自作QR法による固有値結果 (NumPy未使用) ---")
    numpy_elapsed_time = 0.0

    # --- 3. CSVへの結果保存 ---
    output_file = "koyuti_qr_results.csv"
    write_result_to_csv(
        output_file,
        [
            {
                "Method": "Custom QR Method",
                "Matrix Size": matrix_size,
                "Elapsed Time (s)": qr_elapsed_time,
                "Eigenvalues": rounded_custom_eigenvalues,
            }
        ],
    )
    print(f"結果を '{output_file}' に書き込みました。 (NumPy未使用)")


if __name__ == "__main__":
    main()