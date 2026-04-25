import numpy as np
import matplotlib.pyplot as plt


def generar_laberinto(L, l, n, grosor=4):
    muro = 5
    A = np.zeros((L, L))

    # Paredes Perimetrales
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
                # Usamos j+l:j+l+grosor para ensanchar la pared
                # Sumamos 'grosor' también en 'i' para que las esquinas conecten bien
                A[i:i+l+grosor, j+l:j+l+grosor] = 1 
            
            # Pared Superior (ahora con grosor)
            if n2 > n:
                A[i+l:i+l+grosor, j:j+l+grosor] = 1
                
    return A

def generar_laberinto_1():
    dimx = 501
    canal = 55
    aa = np.zeros((dimx, dimx))
    
    # Python empieza en 0. MATLAB i=canal+1 -> Python i=canal
    for i in range(canal, dimx - canal, canal):
        for j in range(0, dimx - canal, canal):
            # i-1:i+1 en MATLAB son 3 celdas. En Python es i-1:i+2
            aa[i-1:i+2, j:j+canal] = np.round(0.25 + np.random.rand())
            
    # Bordes (1:3 en MATLAB son 3 de grosor. En Python es :3)
    aa[:, :3] = 1
    aa[:, -3:] = 1
    aa[:3, :] = 1
    aa[-3:, :-canal] = 1
    
    # Guardar y plotear
    #np.savetxt('laberinto.dat', aa, fmt='%d')
    return aa

def generar_laberinto_2():
    dimx = 501
    canal = 55
    extra = 5
    peso = -0.30
    aa = np.zeros((dimx, dimx))
    
    for i in range(0, dimx - canal, canal):
        for j in range(0, dimx - canal, canal):
            if np.round(peso + np.random.rand()) == 1:
                aa[i:i+3, j:j+canal+extra] = 1
            if np.round(peso + np.random.rand()) == 1:
                aa[i:i+canal+extra, j:j+3] = 1
                
    aa[:, :3] = 1
    aa[:, -3:] = 1
    aa[:3, :] = 1
    aa[-3:, :-canal] = 1
    
    #np.savetxt('laberinto_301x301_complicado.dat', aa, fmt='%d')

    return aa

def generar_laberinto_3():
    dimx = 501
    canal = 65
    extra = 0
    peso = -0.2
    aa = np.zeros((dimx, dimx))
    
    numerosi = np.zeros((dimx, dimx))
    numerosd = np.zeros((dimx, dimx))
    numerosr = np.zeros((dimx, dimx))
    numerosb = np.zeros((dimx, dimx))
    
    for i in range(0, dimx - canal, canal):
        for j in range(0, dimx - canal, canal):
            contador = 6
            while contador > 2:
                numerosi[i, j] = np.round(peso + np.random.rand())
                numerosd[i, j] = np.round(peso + np.random.rand())
                numerosr[i, j] = np.round(peso + np.random.rand())
                numerosb[i, j] = np.round(peso + np.random.rand())
                contador = numerosi[i, j] + numerosd[i, j] + numerosr[i, j] + numerosb[i, j]
                
    for i in range(0, dimx - canal, canal):
        for j in range(0, dimx - canal, canal):
            if numerosi[i, j] == 1:
                aa[i:i+4, j:j+canal+extra] = 1
            if numerosb[i, j] == 1:
                aa[i:i+canal+extra, j:j+4] = 1
            if numerosd[i, j] == 1:
                aa[i+canal:i+canal+4, j:j+canal+extra] = 1
            if numerosr[i, j] == 1:
                aa[i:i+canal+extra, j+canal:j+canal+4] = 1
                
    aa[:, :3] = 1
    aa[:, -3:] = 1
    aa[:3, :] = 1
    aa[-3:, :] = 1
    
    #np.savetxt('laberinto_1003x1003_10.dat', aa, fmt='%d')
    return aa

def generar_laberinto_4():
    dimx = 501
    canal = 50
    extra = 1
    peso = 0.50
    aa = np.zeros((dimx, dimx))
    
    for i in range(0, dimx - 2*canal, 2*canal):
        for j in range(0, dimx - 2*canal, 2*canal):
            # En el original usa round(peso+2*rand) para sacar 0, 1 o 2.
            rand_val = np.round(peso + 2 * np.random.rand())
            if rand_val == 1:
                aa[i:i+3, j:j+2*canal+extra] = 1
                aa[i+canal:i+3+canal, j:j+2*canal+extra] = 1
            elif rand_val == 2:
                aa[i:i+2*canal+extra, j:j+3] = 1
                aa[i:i+2*canal+extra, j+canal:j+3+canal] = 1
                
    aa[:, :3] = 1
    aa[:, -3:] = 1
    aa[:3, :] = 1
    aa[-3:, :] = 1
    
    #np.savetxt('laberinto2_601x601.dat', aa, fmt='%d')
    return aa



if __name__ == '__main__':
    #lab = generar_laberinto(L=501, l=75, n=0.67, grosor=4)
    lab = generar_laberinto_3()
    plt.figure(figsize=(6,6))
    plt.pcolormesh(lab, cmap='binary')
    plt.axis('off')
    plt.show()