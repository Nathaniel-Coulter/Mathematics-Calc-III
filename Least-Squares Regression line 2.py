import numpy as np

def least_squares_regression(x, y):
    
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)  

    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x ** 2)

    
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / n
    
    return m, b, sum_x, sum_y, sum_xy, sum_x_squared, n

def partial_derivatives(n, sum_x, sum_y, sum_xy, sum_x_squared):
    
    fm = (2 * (sum_xy - (sum_y * sum_x) / n)) / n
    fb = (2 * (sum_y - (sum_x * sum_y) / n) - (2 * n * (sum_y - sum_y / n))) / n
    
    return fm, fb

def critical_points(n, sum_x, sum_y, sum_xy, sum_x_squared):
    
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / n
    return m, b

def hessian_matrix(n, sum_x, sum_x_squared):
    
    f_mm = 2 * sum_x_squared
    f_bb = 2 * n
    f_mb = -2 * np.sum(x_data)  
    
    return f_mm, f_bb, f_mb

def main():
    
    global x_data  
    x_data = [1, 2, 3, 4, 5]
    y_data = [2, 3, 5, 7, 11]

    
    m, b, sum_x, sum_y, sum_xy, sum_x_squared, n = least_squares_regression(x_data, y_data)

    
    fm, fb = partial_derivatives(n, sum_x, sum_y, sum_xy, sum_x_squared)
    
   
    critical_m, critical_b = critical_points(n, sum_x, sum_y, sum_xy, sum_x_squared)

    
    f_mm, f_bb, f_mb = hessian_matrix(n, sum_x, sum_x_squared)
    
    
    determinant = f_mm * f_bb - f_mb ** 2

    
    print(f"Least-Squares Regression Line: y = {m:.4f}x + {b:.4f}\n")
    print(f"Partial Derivative with respect to m (fm): {fm:.4f}")
    print(f"Partial Derivative with respect to b (fb): {fb:.4f}\n")
    print(f"Critical Point for m: {critical_m:.4f}")
    print(f"Critical Point for b: {critical_b:.4f}\n")
    
    print("Hessian Matrix:")
    print(f"f_mm: {f_mm:.4f}, f_bb: {f_bb:.4f}, f_mb: {f_mb:.4f}\n")
    
    print(f"Determinant of Hessian: {determinant:.4f}\n")
    
    
    if determinant > 0 and f_mm > 0:
        print("The critical point is a local minimum.")
    else:
        print("The critical point is not a local minimum.")

if __name__ == "__main__":
    main()
    