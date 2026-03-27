import numpy as np
import matplotlib.pyplot as plt

def generar_laberinto(L=203, l=25, n=0.67):
    muro = 3
    A = np.zeros((L, L))

    # Paredes Perimetrales (Usando "slicing" de NumPy, más rápido que bucles)
    A[:, :muro] = 1        # Izquierda
    A[L-muro:, :] = 1      # Superior
    A[:, L-muro:] = 1      # Derecha
    A[:muro, :] = 1        # Inferior

    # Laberinto: Recorrido por celdas
    for i in range(0, L - l, l):
        for j in range(0, L - l, l):
            n1 = np.random.rand()
            n2 = np.random.rand()
            
            # Pared Derecha
            if n1 > n:
                A[i:i+l+1, j+l] = 1 # El +1 es para cerrar esquinas
            
            # Pared Superior
            if n2 > n:
                A[i+l, j:j+l+1] = 1
                
    return A

# Generar y mostrar
lab = generar_laberinto()
plt.figure(figsize=(8,8))
plt.pcolormesh(lab, cmap='binary')
plt.axis('off')
plt.show()