import numpy as np

def least_squares_regression(x, y):
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)  # Number of data points

    # Calculate necessary sums
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x ** 2)

    # Calculate slope (m) and intercept (b)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / n
    
    return m, b, sum_x, sum_y, sum_xy, sum_x_squared, n

def partial_derivatives(n, sum_x, sum_y, sum_xy, sum_x_squared):
    # Partial derivatives
    fm = (2 * (sum_xy - (sum_y * sum_x) / n)) / n
    fb = (2 * (sum_y - (sum_x * sum_y) / n) - (2 * n * (sum_y - sum_y / n))) / n
    
    return fm, fb

def critical_points(n, sum_x, sum_y, sum_xy, sum_x_squared):
    # Calculate critical points for m and b
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / n
    return m, b

def hessian_matrix(n, sum_x, sum_x_squared):
    # Calculate the elements of the Hessian matrix
    f_mm = 2 * sum_x_squared
    f_bb = 2 * n
    f_mb = -2 * np.sum(x_data)  # Using the global variable x_data
    
    return f_mm, f_bb, f_mb

def main():
    # Example data points
    global x_data  # Declare x_data as global to use in hessian_matrix
    x_data = [1, 2, 3, 4, 5]
    y_data = [2, 3, 5, 7, 11]

    # Calculate the least-squares regression line and necessary sums
    m, b, sum_x, sum_y, sum_xy, sum_x_squared, n = least_squares_regression(x_data, y_data)

    # Calculate partial derivatives
    fm, fb = partial_derivatives(n, sum_x, sum_y, sum_xy, sum_x_squared)
    
    # Calculate critical points
    critical_m, critical_b = critical_points(n, sum_x, sum_y, sum_xy, sum_x_squared)

    # Calculate Hessian matrix
    f_mm, f_bb, f_mb = hessian_matrix(n, sum_x, sum_x_squared)
    
    # Calculate determinant of Hessian
    determinant = f_mm * f_bb - f_mb ** 2

    # Output results with explanations
    print(f"Least-Squares Regression Line: y = {m:.4f}x + {b:.4f}\n")
    print(f"Partial Derivative with respect to m (fm): {fm:.4f}")
    print(f"Partial Derivative with respect to b (fb): {fb:.4f}\n")
    print(f"Critical Point for m: {critical_m:.4f}")
    print(f"Critical Point for b: {critical_b:.4f}\n")
    
    print("Hessian Matrix:")
    print(f"f_mm: {f_mm:.4f}, f_bb: {f_bb:.4f}, f_mb: {f_mb:.4f}\n")
    
    print(f"Determinant of Hessian: {determinant:.4f}\n")
    
    # Second Derivative Test
    if determinant > 0 and f_mm > 0:
        print("The critical point is a local minimum.")
    else:
        print("The critical point is not a local minimum.")

if __name__ == "__main__":
    main()
    