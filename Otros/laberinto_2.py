import numpy as np
import matplotlib.pyplot as plt

def generar_laberinto_2(dimx=301, canal=30, peso=-0.30):
    extra = 1
    # Inicializamos la matriz de ceros (Python usa 0 por defecto)
    aa = np.zeros((dimx, dimx))
    
    # Generación de paredes internas
    # np.arange(inicio, fin, paso) -> el fin no se incluye
    for i in range(0, dimx - canal, canal):
        for j in range(0, dimx - canal, canal):
            # Lógica: round(peso + rand) == 1
            if np.round(peso + np.random.rand()) == 1:
                aa[i:i+3, j:j+canal+extra] = 1
            
            if np.round(peso + np.random.rand()) == 1:
                aa[i:i+canal+extra, j:j+3] = 1

    # Paredes perimetrales (Ajuste de índices para Python)
    aa[:, :3] = 1              # Izquierda
    aa[:, -3:] = 1             # Derecha
    aa[:3, :] = 1              # Superior
    aa[-3:, :dimx-canal] = 1   # Inferior con hueco de salida
    
    return aa

# Visualización
lab1 = generar_laberinto_2()
plt.figure(figsize=(8,8))
plt.pcolormesh(lab1, cmap='Greys') # pcolor equivalente
plt.axis('equal')
plt.show()

# Para guardar (equivalente a -ASCII)
np.savetxt("laberinto_python.dat", lab1, fmt='%d')