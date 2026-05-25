import csv
import argparse
import random
import time

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


def validate_section_3_1():
    print("3.1 3次正方行列を用いたアルゴリズムの正当性検証")

    determinant_matrix = [[1, 2, 3], [0, 4, 5], [1, 0, 2]]
    determinant_expected = 6
    determinant_value, determinant_time = determinant_with_time(determinant_matrix)
    print("\n(1) 行列式の検証")
    print(f"入力行列: {determinant_matrix}")
    print(f"既知の解: {determinant_expected}")
    print(f"計算結果: {round(determinant_value, 6)}")
    print(f"一致: {abs(determinant_value - determinant_expected) < 1e-9}")
    print(f"計算時間: {determinant_time:.6f} 秒")

    linear_matrix = [[1, 1, 1, 2], [2, -1, 1, 7], [1, 2, -1, -6]]
    linear_expected = [1, -2, 3]
    linear_start_time = time.perf_counter()
    linear_solution = solve_linear_equation(linear_matrix)
    linear_time = time.perf_counter() - linear_start_time
    print("\n(2) 連立1次方程式の検証（拡大係数行列）")
    print(f"入力行列: {linear_matrix}")
    print(f"既知の解: x = {linear_expected[0]}, y = {linear_expected[1]}, z = {linear_expected[2]}")
    print(f"計算結果: {rounded_list(linear_solution) if linear_solution is not None else linear_solution}")
    print(f"一致: {linear_solution is not None and all(abs(linear_solution[i] - linear_expected[i]) < 1e-9 for i in range(3))}")
    print(f"計算時間: {linear_time:.6f} 秒")

    eigen_matrix = [[2, 1, 1], [0, 3, 2], [0, 0, 4]]
    eigen_expected = [2, 3, 4]
    eigen_start_time = time.perf_counter()
    # Use QR method (pure Python) to get eigenvalue approximations for verification
    eigen_values = qr_method(eigen_matrix)
    eigen_time = time.perf_counter() - eigen_start_time
    print("\n(3) 固有値の検証 (QR法による近似)")
    print(f"入力行列: {eigen_matrix}")
    print("既知の解:")
    for value in eigen_expected:
        print(f"固有値: {value}")
    print("計算結果 (QR近似):")
    for value in eigen_values:
        print(f"固有値: {round(value, 6)}")
    print(f"計算時間: {eigen_time:.6f} 秒")


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
            "Custom Linear Time (s)",
            "Custom Dominant Eigenvalue",
            "Custom Eigen Time (s)",
            "RHS",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "Matrix Size": row.get("Matrix Size"),
                    "Custom Determinant Time (s)": row.get("Custom Determinant Time (s)"),
                    "Custom Linear Time (s)": row.get("Custom Linear Time (s)"),
                    "Custom Dominant Eigenvalue": row.get("Custom Dominant Eigenvalue"),
                    "Custom Eigen Time (s)": row.get("Custom Eigen Time (s)"),
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