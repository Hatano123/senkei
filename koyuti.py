import numpy as np
import csv
import time


def power_method(matrix, iterations=100):
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
        for i in range(size):
            if abs(next_vector[i]) > abs(eigenvalue):
                eigenvalue = next_vector[i]

        for i in range(size):
            vector[i] = next_vector[i] / eigenvalue

    return eigenvalue, vector


def write_result_to_csv(output_file, rows):
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "Method",
            "Matrix Size",
            "Elapsed Time (s)",
            "Max Eigenvalue",
            "Eigenvector",
            "Eigenvalues",
            "Eigenvectors",
            "Matrix",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    matrix_size = int(input("行列の次数Nを入力: "))
    matrix = [list(map(float, input().split())) for _ in range(matrix_size)]

    power_start_time = time.perf_counter()
    max_eigenvalue, power_vector = power_method(matrix)
    power_elapsed_time = time.perf_counter() - power_start_time

    print(f"\n最大の固有値: {round(max_eigenvalue, 5)}")
    rounded_vector = []
    for value in power_vector:
        rounded_vector.append(round(value, 5))
    print(f"対応する固有ベクトル: {rounded_vector}")
    print(f"べき乗法の計算時間: {power_elapsed_time:.6f} 秒")

    numpy_start_time = time.perf_counter()
    numpy_matrix = np.array(matrix)
    eigenvalues, eigenvectors = np.linalg.eig(numpy_matrix)
    numpy_elapsed_time = time.perf_counter() - numpy_start_time

    print("\n--- NumPyによる計算結果 ---")
    for i in range(matrix_size):
        val = round(eigenvalues[i], 5)
        print(f"固有値{i+1}: {val}")

        vec = []
        for j in range(matrix_size):
            vec.append(round(eigenvectors[j][i], 5))
        print(f"固有ベクトル{i+1}: {vec}\n")

    print(f"NumPyの計算時間: {numpy_elapsed_time:.6f} 秒")

    output_file = "koyuti_results.csv"
    write_result_to_csv(
        output_file,
        [
            {
                "Method": "Power Method",
                "Matrix Size": matrix_size,
                "Elapsed Time (s)": power_elapsed_time,
                "Max Eigenvalue": max_eigenvalue,
                "Eigenvector": rounded_vector,
                "Eigenvalues": "",
                "Eigenvectors": "",
                "Matrix": matrix,
            },
            {
                "Method": "NumPy eig",
                "Matrix Size": matrix_size,
                "Elapsed Time (s)": numpy_elapsed_time,
                "Max Eigenvalue": "",
                "Eigenvector": "",
                "Eigenvalues": eigenvalues.tolist(),
                "Eigenvectors": eigenvectors.tolist(),
                "Matrix": matrix,
            },
        ],
    )
    print(f"結果を '{output_file}' に書き込みました。")


if __name__ == "__main__":
    main()