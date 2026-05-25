import csv
import time


def solve_linear_equation(matrix):
    working_matrix = [row[:] for row in matrix]

    for i in range(len(working_matrix)):
        if working_matrix[i][i] == 0:
            for k in range(i + 1, len(working_matrix)):
                if working_matrix[k][i] != 0:
                    working_matrix[i], working_matrix[k] = working_matrix[k], working_matrix[i]
                    break
            else:
                return None

        pivot = working_matrix[i][i]
        if pivot == 0:
            return None

        for j in range(i, len(working_matrix) + 1):
            working_matrix[i][j] /= pivot

        for k in range(len(working_matrix)):
            if k != i:
                factor = working_matrix[k][i]
                for j in range(i, len(working_matrix) + 1):
                    working_matrix[k][j] -= factor * working_matrix[i][j]

    return [working_matrix[i][len(working_matrix)] for i in range(len(working_matrix))]


def write_result_to_csv(output_file, matrix_size, matrix, solution, elapsed_time):
    # Reduce output: do not include full matrix in CSV to prevent large files
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["Matrix Size", "Elapsed Time (s)", "Solution"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "Matrix Size": matrix_size,
                "Elapsed Time (s)": elapsed_time,
                "Solution": solution,
            }
        )


def main():
    matrix_size = int(input("変数の数Nを入力: "))
    matrix = [list(map(float, input().split())) for _ in range(matrix_size)]

    start_time = time.perf_counter()
    solution = solve_linear_equation(matrix)
    elapsed_time = time.perf_counter() - start_time

    if solution is None:
        print("解が一意に定まりませんでした。")
    else:
        print("解:")
        for i in range(matrix_size):
            print(f"x{i+1} = {solution[i]}")

    print(f"計算時間: {elapsed_time:.6f} 秒")

    output_file = "renritu_results.csv"
    write_result_to_csv(output_file, matrix_size, matrix, solution, elapsed_time)
    print(f"結果を '{output_file}' に書き込みました。")


if __name__ == "__main__":
    main()