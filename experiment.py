import csv
import argparse
import random
import time
import math
import numpy as np

from gyoretushiki import determinant_with_time
from koyuti import power_method, to_integer_vector, qr_method
from renritu import solve_linear_equation


def rounded_list(values, digits=6):
    return [round(float(value), digits) for value in values]


def generate_random_matrix(size, lower=0, upper=9):
    return [[random.randint(lower, upper) for _ in range(size)] for _ in range(size)]


def generate_invertible_matrix(size, lower=0, upper=9):
    while True:
        matrix = generate_random_matrix(size, lower, upper)
        det_value, _ = determinant_with_time(matrix)
        if abs(det_value) > 1e-9:
            return matrix


def normalized_direction_error(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return float('nan')

    a = np.asarray(vec_a, dtype=np.complex128).reshape(-1)
    b = np.asarray(vec_b, dtype=np.complex128).reshape(-1)

    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float('nan')

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return float('nan')

    a_unit = a / a_norm
    b_unit = b / b_norm

    same_direction = np.linalg.norm(a_unit - b_unit)
    opposite_direction = np.linalg.norm(a_unit + b_unit)
    return float(min(same_direction, opposite_direction))


def validate_section_3_1():
    print("3.1 3次正方行列を用いたアルゴリズムの正当性検証")

    determinant_matrix = [[1, 2, 3], [0, 4, 5], [1, 0, 2]]
    determinant_expected = 6
    determinant_value, determinant_time = determinant_with_time(determinant_matrix)
    # NumPy determinant for comparison
    np_det_start = time.perf_counter()
    np_det_value = float(np.linalg.det(np.array(determinant_matrix, dtype=float)))
    np_det_time = time.perf_counter() - np_det_start
    det_error = abs(determinant_value - np_det_value)
    print("\n(1) 行列式の検証")
    print(f"入力行列: {determinant_matrix}")
    print(f"既知の解: {determinant_expected}")
    print(f"計算結果: {round(determinant_value, 6)}")
    print(f"一致: {abs(determinant_value - determinant_expected) < 1e-9}")
    print(f"計算時間 (自作): {determinant_time:.6f} 秒")
    print(f"NumPy 計算結果: {round(np_det_value,6)}")
    print(f"NumPy 計算時間: {np_det_time:.6f} 秒")
    print(f"自作と NumPy の差: {det_error}")

    linear_matrix = [[1, 1, 1, 2], [2, -1, 1, 7], [1, 2, -1, -6]]
    linear_expected = [1, -2, 3]
    linear_start_time = time.perf_counter()
    linear_solution = solve_linear_equation(linear_matrix)
    linear_time = time.perf_counter() - linear_start_time
    # NumPy solve for comparison
    A = [row[:-1] for row in linear_matrix]
    b = [row[-1] for row in linear_matrix]
    np_linear_start = time.perf_counter()
    np_linear_solution = list(np.linalg.solve(np.array(A, dtype=float), np.array(b, dtype=float)))
    np_linear_time = time.perf_counter() - np_linear_start
    linear_error = math.sqrt(sum((linear_solution[i] - np_linear_solution[i]) ** 2 for i in range(3))) if linear_solution is not None else float('nan')
    print("\n(2) 連立1次方程式の検証（拡大係数行列）")
    print(f"入力行列: {linear_matrix}")
    print(f"既知の解: x = {linear_expected[0]}, y = {linear_expected[1]}, z = {linear_expected[2]}")
    print(f"計算結果: {rounded_list(linear_solution) if linear_solution is not None else linear_solution}")
    print(f"一致: {linear_solution is not None and all(abs(linear_solution[i] - linear_expected[i]) < 1e-9 for i in range(3))}")
    print(f"計算時間 (自作): {linear_time:.6f} 秒")
    print(f"NumPy 計算結果: {rounded_list(np_linear_solution)}")
    print(f"NumPy 計算時間: {np_linear_time:.6f} 秒")
    print(f"自作と NumPy の L2 ノルム差: {linear_error}")

    eigen_matrix = [[2, 1, 1], [0, 3, 2], [0, 0, 4]]
    eigen_expected = [2, 3, 4]
    eigen_start_time = time.perf_counter()
    # Use QR method (pure Python) to get eigenvalue approximations for verification
    eigen_values = qr_method(eigen_matrix)
    eigen_time = time.perf_counter() - eigen_start_time
    # NumPy eigenvalues for comparison
    np_eigen_start = time.perf_counter()
    np_eigvals = list(np.linalg.eigvals(np.array(eigen_matrix, dtype=float)))
    np_eigen_time = time.perf_counter() - np_eigen_start
    print("\n(3) 固有値の検証 (QR法による近似)")
    print(f"入力行列: {eigen_matrix}")
    print("既知の解:")
    for value in eigen_expected:
        print(f"固有値: {value}")
    print("計算結果 (QR近似):")
    for value in eigen_values:
        print(f"固有値: {round(value, 6)}")
    print(f"計算時間 (QR, 自作): {eigen_time:.6f} 秒")
    print(f"NumPy 固有値: {[round(v,6) for v in np_eigvals]}")
    print(f"NumPy 計算時間: {np_eigen_time:.6f} 秒")


def benchmark_section_3_2(max_size=10, seed=42):
    print("\n3.2 n次正方行列を用いた大規模計算の性能検証")
    print("要素が0から9のランダムな整数で構成される n 次正方行列を自動生成し、自作プログラムの性能を計測します。")

    random.seed(seed)
    results = []

    for size in range(1, max_size + 1):
        matrix = generate_invertible_matrix(size)
        rhs = [random.randint(0, 9) for _ in range(size)]
        augmented_matrix = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]

        custom_det_value, custom_det_time = determinant_with_time(matrix)

        custom_linear_start = time.perf_counter()
        custom_linear_solution = solve_linear_equation(augmented_matrix)
        custom_linear_time = time.perf_counter() - custom_linear_start

        custom_eigen_start = time.perf_counter()
        custom_eigenvalue, custom_eigenvector = power_method(matrix)
        custom_eigen_time = time.perf_counter() - custom_eigen_start
        custom_integer_eigenvector = to_integer_vector(custom_eigenvector)

        # NumPy comparisons
        np_matrix = np.array(matrix, dtype=float)
        np_rhs = np.array(rhs, dtype=float)

        np_det_start = time.perf_counter()
        try:
            np_det_value = float(np.linalg.det(np_matrix))
        except Exception:
            np_det_value = float('nan')
        np_det_time = time.perf_counter() - np_det_start

        np_linear_start = time.perf_counter()
        try:
            np_linear_solution = list(np.linalg.solve(np_matrix, np_rhs))
        except Exception:
            np_linear_solution = None
        np_linear_time = time.perf_counter() - np_linear_start

        np_eigen_start = time.perf_counter()
        try:
            eigvals, eigvecs = np.linalg.eig(np_matrix)
            idx = int(np.argmax(np.abs(eigvals)))
            np_eigenvalue = float(eigvals[idx])
            np_eigenvector = list(eigvecs[:, idx])
        except Exception:
            np_eigenvalue = float('nan')
            np_eigenvector = None
        np_eigen_time = time.perf_counter() - np_eigen_start

        # Error / accuracy metrics
        def diff_norm(a, b):
            if a is None or b is None:
                return float('nan')
            return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(len(a))))

        det_error = None if (custom_det_value is None or np_det_value is None) else abs(custom_det_value - np_det_value)
        linear_error = diff_norm(custom_linear_solution, np_linear_solution) if (custom_linear_solution is not None and np_linear_solution is not None) else float('nan')
        eigenvalue_error = float('nan') if (custom_eigenvalue is None or np_eigenvalue is None) else abs(custom_eigenvalue - np_eigenvalue)
        eigenvector_error = normalized_direction_error(custom_eigenvector, np_eigenvector)

        results.append(
            {
                "Matrix Size": size,
                "Custom Determinant": custom_det_value,
                "Custom Determinant Time (s)": custom_det_time,
                "Custom Linear Solution": custom_linear_solution,
                "Custom Linear Time (s)": custom_linear_time,
                "Custom Dominant Eigenvalue": custom_eigenvalue,
                "Custom Eigen Time (s)": custom_eigen_time,
                "Custom Dominant Eigenvector": custom_integer_eigenvector,
                "NumPy Determinant": np_det_value,
                "NumPy Determinant Time (s)": np_det_time,
                "NumPy Linear Solution": np_linear_solution,
                "NumPy Linear Time (s)": np_linear_time,
                "NumPy Dominant Eigenvalue": np_eigenvalue,
                "NumPy Eigen Time (s)": np_eigen_time,
                "NumPy Dominant Eigenvector": np_eigenvector,
                "Determinant Error": det_error,
                "Linear Solution L2 Error": linear_error,
                "Eigenvalue Error": eigenvalue_error,
                "Eigenvector L2 Error": eigenvector_error,
                "Matrix": None,
                "RHS": rhs,
            }
        )

        print(f"\n次数 n = {size}")
        print(f"行列式: 自作={round(custom_det_value, 6)}")
        print(f"連立方程式: 自作={rounded_list(custom_linear_solution) if custom_linear_solution is not None else custom_linear_solution}")
        print(f"最大固有値: 自作={round(custom_eigenvalue, 6)}")

    # Write a reduced set of fields to keep CSV small (no full matrices/vectors)
    # Write reduced CSV with only custom-method summaries
    with open("experiment_results.csv", "w", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = [
            "Matrix Size",
            "Custom Determinant Time (s)",
            "NumPy Determinant Time (s)",
            "Determinant Error",
            "Custom Linear Time (s)",
            "NumPy Linear Time (s)",
            "Linear Solution L2 Error",
            "Custom Eigen Time (s)",
            "NumPy Eigen Time (s)",
            "Eigenvalue Error",
            "Eigenvector L2 Error",
            "Custom Dominant Eigenvalue",
            "NumPy Dominant Eigenvalue",
            "RHS",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "Matrix Size": row.get("Matrix Size"),
                    "Custom Determinant Time (s)": row.get("Custom Determinant Time (s)"),
                    "NumPy Determinant Time (s)": row.get("NumPy Determinant Time (s)"),
                    "Determinant Error": row.get("Determinant Error"),
                    "Custom Linear Time (s)": row.get("Custom Linear Time (s)"),
                    "NumPy Linear Time (s)": row.get("NumPy Linear Time (s)"),
                    "Linear Solution L2 Error": row.get("Linear Solution L2 Error"),
                    "Custom Eigen Time (s)": row.get("Custom Eigen Time (s)"),
                    "NumPy Eigen Time (s)": row.get("NumPy Eigen Time (s)"),
                    "Eigenvalue Error": row.get("Eigenvalue Error"),
                    "Eigenvector L2 Error": row.get("Eigenvector L2 Error"),
                    "Custom Dominant Eigenvalue": row.get("Custom Dominant Eigenvalue"),
                    "NumPy Dominant Eigenvalue": row.get("NumPy Dominant Eigenvalue"),
                    "RHS": row.get("RHS"),
                }
            )

    print("\n結果を 'experiment_results.csv' に書き込みました。")


def main():
    parser = argparse.ArgumentParser(description="3.1/3.2 の検証・性能計測を実行します。")
    parser.add_argument(
        "--mode",
        choices=["all", "validate", "benchmark"],
        default="all",
        help="実行モードを指定します。",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=10,
        help="benchmark モードで計測する最大次数 n を指定します。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シードを指定します。",
    )
    args = parser.parse_args()

    if args.mode in ("all", "validate"):
        validate_section_3_1()

    if args.mode in ("all", "benchmark"):
        benchmark_section_3_2(max_size=args.max_size, seed=args.seed)


if __name__ == "__main__":
    main()