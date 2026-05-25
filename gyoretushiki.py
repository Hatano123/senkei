import csv
import time


def determinant_with_time(matrix):
    working_matrix = [row[:] for row in matrix]
    determinant = 1.0

    start_time = time.perf_counter()

    for i in range(len(working_matrix)):
        if working_matrix[i][i] == 0:
            for k in range(i + 1, len(working_matrix)):
                if working_matrix[k][i] != 0:
                    working_matrix[i], working_matrix[k] = working_matrix[k], working_matrix[i]
                    determinant *= -1
                    break
            else:
                determinant = 0.0
                break

        determinant *= working_matrix[i][i]

        for k in range(i + 1, len(working_matrix)):
            factor = working_matrix[k][i] / working_matrix[i][i]
            for j in range(i, len(working_matrix)):
                working_matrix[k][j] -= factor * working_matrix[i][j]

    elapsed_time = time.perf_counter() - start_time
    return determinant, elapsed_time


def write_result_to_csv(output_file, matrix_size, matrix, determinant, elapsed_time):
    # Reduce output: do not write full matrix to CSV to keep file small
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["Matrix Size", "Elapsed Time (s)", "Determinant"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "Matrix Size": matrix_size,
            "Elapsed Time (s)": elapsed_time,
            "Determinant": determinant,
        })


def main():
    matrix_size = int(input("行列の次数Nを入力: "))
    matrix = [list(map(float, input().split())) for _ in range(matrix_size)]

    determinant, elapsed_time = determinant_with_time(matrix)

    print(f"|A| = {determinant}")
    print(f"計算時間: {elapsed_time:.6f} 秒")

    output_file = "gyoretushiki_results.csv"
    write_result_to_csv(output_file, matrix_size, matrix, determinant, elapsed_time)
    print(f"結果を '{output_file}' に書き込みました。")


if __name__ == "__main__":
    main()