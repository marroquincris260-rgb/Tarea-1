# ==========================================
# METODO DE GAUSS-JORDAN (ADAPTADO)
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

def gauss_jordan_pivot_determinante(A, b):
    n = len(A)
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(float)

    det_A = np.linalg.det(A)

    if np.isclose(det_A, 0):
        print("Sistema sin solución única")
        return None

    print("Determinante:", det_A)

    for i in range(n):
        # Pivoteo
        max_row = np.argmax(abs(Ab[i:, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        # Normalizar pivote
        Ab[i] = Ab[i] / Ab[i, i]

        # Eliminar otras filas
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[j, i] * Ab[i]

    return Ab[:, -1]

# ------------------------------------------
# ERROR
# ------------------------------------------
def calcular_error(A, x, b):
    return np.dot(A, x) - b

# ------------------------------------------
# GRAFICA
# ------------------------------------------
def graficar_error(error, titulo):
    plt.figure()
    plt.plot(error, marker='o')
    plt.title(titulo)
    plt.xlabel("Ecuación")
    plt.ylabel("Error")
    plt.grid()
    plt.show()

# ==========================================
# EJERCICIO 1 (8x8)
# ==========================================

A1 = np.array([
[2,3,-1,4,-2,5,-3,1],
[-3,2,4,-1,3,-2,5,-1],
[4,-1,3,2,-3,1,-2,5],
[-1,5,-2,3,4,-1,2,-3],
[3,-2,5,-1,4,2,-3,1],
[-2,4,-3,1,5,-1,2,-4],
[5,-1,2,-3,4,1,-2,3],
[1,-3,4,-2,5,-1,2,-1]
], dtype=float)

b1 = np.array([10,-5,8,4,-7,6,-3,9], dtype=float)

sol1 = gauss_jordan_pivot_determinante(A1.copy(), b1.copy())

if sol1 is not None:
    error1 = calcular_error(A1, sol1, b1)
    print("\nEjercicio 1 Solución:\n", sol1)
    print("Error:\n", error1)
    graficar_error(error1, "Error Ejercicio 1")

# ==========================================
# EJERCICIO 2 (9x9)
# ==========================================

A2 = np.array([
[3,-2,5,-1,4,2,-3,1,2],
[-2,4,-3,1,5,-1,2,-4,3],
[5,-1,2,-3,4,1,-2,3,-1],
[1,-3,4,-2,5,-1,2,-1,4],
[2,3,-1,4,-2,5,-3,1,-2],
[-3,2,4,-1,3,-2,5,-1,1],
[4,-1,3,2,-3,1,-2,5,-4],
[-1,5,-2,3,4,-1,2,-3,1],
[3,-2,5,-1,4,2,-3,1,-5]
], dtype=float)

b2 = np.array([-8,7,-6,5,12,-9,10,3,-2], dtype=float)

sol2 = gauss_jordan_pivot_determinante(A2.copy(), b2.copy())

if sol2 is not None:
    error2 = calcular_error(A2, sol2, b2)
    print("\nEjercicio 2 Solución:\n", sol2)
    print("Error:\n", error2)
    graficar_error(error2, "Error Ejercicio 2")

# ==========================================
# EJERCICIO 3 (10x10)
# ==========================================

A3 = np.array([
[2,-3,4,-1,5,-1,2,-1,3,-2],
[-3,2,5,-1,4,2,-3,1,-2,5],
[4,-1,3,2,-3,1,-2,5,-4,1],
[-1,5,-2,3,4,-1,2,-3,1,-5],
[3,-2,5,-1,4,2,-3,1,-5,2],
[-2,4,-3,1,5,-1,2,-4,3,-1],
[5,-1,2,-3,4,1,-2,3,-1,4],
[1,-3,4,-2,5,-1,2,-1,4,-3],
[2,3,-1,4,-2,5,-3,1,-2,1],
[-3,2,4,-1,3,-2,5,-1,1,-4]
], dtype=float)

b3 = np.array([11,-10,8,-6,7,-3,9,-5,6,-8], dtype=float)

sol3 = gauss_jordan_pivot_determinante(A3.copy(), b3.copy())

if sol3 is not None:
    error3 = calcular_error(A3, sol3, b3)
    print("\nEjercicio 3 Solución:\n", sol3)
    print("Error:\n", error3)
    graficar_error(error3, "Error Ejercicio 3")
