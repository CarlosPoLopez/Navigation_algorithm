# Resolución de Laberintos mediante Sistemas de Reacción-Difusión (Modelo FitzHugh-Nagumo)

Este proyecto implementa una simulación numérica en Python del modelo de reacción-difusión de **FitzHugh-Nagumo (FHN)**. El sistema utiliza la interacción entre un activador y un inhibidor para inundar un laberinto y, posteriormente, identificar el camino más corto mediante un proceso de retracción elástica.

##  Contexto Científico

La simulación resuelve las ecuaciones diferenciales parciales del modelo FHN:
* **Variable $u$ (Activador):** Define el frente de la onda que avanza por los pasillos.
* **Variable $v$ (Inhibidor):** Genera el "escudo repulsor" que evita el colapso de frentes adyacentes, permitiendo que la onda mantenga una estructura central (esqueleto).
* **Matriz $F$:** Representa la geometría del laberinto, actuando como un término de fuerza que apaga la variable activadora en las paredes.

##  Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

* **`main_ida.py`**: Script para la **Fase 1 (Expansión)**. Genera el laberinto y lanza la onda desde el origen hasta la salida.
* **`main_vuelta.py`**: Script para la **Fase 2 (Retracción)**. Carga el estado final de la Fase 1 y calcula el camino óptimo.
* **`bucle.py`**: Motor numérico híbrido. Implementa los métodos de **Dufort-Frankel** (para una expansión estable y rápida) y **FTCS** (para una retracción suave y precisa).
* **`generar_laberinto.py`**: Biblioteca con algoritmos para crear diferentes tipos de obstáculos y geometrías.
* **`OTROS/`**: Carpeta con herramientas de análisis:
    * `nullclines.py`: Estudio de estabilidad del sistema.
    * `onda1.py` / `onda2.py`: Simulaciones de colisión de ondas simples.
    * Datos de soporte técnico.

##  Instalación y Uso

1. **Requisitos**: Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
2. **Fase 1**: Ejecuta el script de ida para generar el laberinto y propagar la autoonda. Se crearán las carpetas `/LAB_ida` (fotogramas) y `/Matrices_ida` (datos .npy para la siguiente fase):
   ```basch
   python main_ida.py
3. **Fase 2**: Una vez finalizada la fase anterior, ejecuta el script de vuelta para hallar el camino más corto. Se crearán las carpetas `/LAB_vuelta` (fotogramas) y `/Matrices_vuelta` (datos .npy):
   ```basch
   python main_vuelta.py
   
   
