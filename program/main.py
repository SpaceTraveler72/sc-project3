import numpy as np

# uses scaled partial pivoting to solve a system of linear equations
def gaussian_elimination(matrix, b, file):
    n = len(matrix)
    scale = [max(map(abs, row)) for row in matrix]  # make an array of scaling factors
    indices = list(range(n))  # Row indices

    file.write(f"Matrix size: {n}x{n}\n\n")  # Write matrix size as a header

    for k in range(n - 1):
        # Scaled partial pivoting
        pivot_row = max(range(k, n), key=lambda i: abs(matrix[indices[i]][k]) / scale[indices[i]])
        indices[k], indices[pivot_row] = indices[pivot_row], indices[k]

        # Perform EROs to eliminate the entries below the pivot
        for i in range(k + 1, n):
            factor = matrix[indices[i]][k] / matrix[indices[k]][k]
            for j in range(k, n):
                matrix[indices[i]][j] -= factor * matrix[indices[k]][j]
            b[indices[i]] -= factor * b[indices[k]]

        # Write intermediate matrix and vector to file and terminal
        step_header = f"Step {k + 1}:\n"
        file.write(step_header)
        print(step_header, end="")
        for idx in indices:
            row = " ".join(f"{matrix[idx][j]:10.3f}" for j in range(n)) + f" | {b[idx]:10.3f}\n"
            file.write(row)
            print(row, end="")
        file.write("\n")
        print()

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):  # Start from the last row and move upwards
        sum_ax = sum(matrix[indices[i]][j] * x[j] for j in range(i + 1, n))  # Sum of known terms
        x[i] = (b[indices[i]] - sum_ax) / matrix[indices[i]][i]  # Solve for x[i]
    return x

def main():
    with open("intermediate_steps.txt", "w") as file:  # Open file in main
        # Use a 3x3 matrix and vector for testing
        matrix = [
            [2, -5, 1],
            [4, 2, 9],
            [-3, 3, 5]
        ]
        b = [2, -1, 4]

        solution = gaussian_elimination(matrix, b, file)
        print("Solution:")
        print(" ".join(f"x{i + 1} = {val:.3f}" for i, val in enumerate(solution)))
        
        # Use the provided matrix and vector
        matrix = [
            [3, -13, 9, 3],
            [-6, 4, 1, -18],
            [6, -2, 2, 4],
            [12, -8, 6, 10]
        ]
        b = [-19, -34, 16, 26]

        solution = gaussian_elimination(matrix, b, file)
        print("Solution:")
        print(" ".join(f"x{i + 1} = {val:.3f}" for i, val in enumerate(solution)))

if __name__ == "__main__":
    main()
