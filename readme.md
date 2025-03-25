# Predicción del siguiente número con CNN 

Este proyecto muestra colmo usar una **Red Neuronal Convolucional (CNN 1D)** para predecir el **siguiente número en una secuencia numerica simple**.

---

## Requisitos

- Python 3.7+
- TensorFlow
- NumPy

---

## Justificacion 

Este modelo esta creado para trabajar con secuencias pequeñas de datos univariantes, es decir, datos que tienen una sola caracteristica por muestra y una longitud temporal de 3 pasos por ejemplo, una secuencia de tres valores: [x1, x2, x3].



### Arquitectura del modelo .

- Referencias:
- Conv1D. (2024, junio). TensorFlow. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
- Sotelo, J. A. L. (2023,68). Deep Learning: teoría y aplicaciones. Marcombo.

model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(3,1)),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1)
])

1. **Capa Convolucinal 1 Dimension**
   - Aplica 32 filtros que son tambien llamados detectores de patrones con un tamaño de kernel de 2 esto examina de a 2 valores dentro de la secuencia.
   - Usa la función de activacion `ReLU` para permitir al modelo aprender relaciones no lineales y pueda aprender patrones mas complejos.
   - Entrada esperada(input_shape): secuencia de 3 pasos de tiempo con 1 caracteristica por paso se selecciono para predicciones simples en cortos periodos de tiempo.

2. **Capa Flatten**
   - Convierte la salida multidimensional de la convolucion en un vector plano.
   - Esto es necesario para conectar con las capas densas.

3. **Capa Densa 16 neuronas**
   - Es una capa densa con 16 neuronas.
   - Aprende combinaciones mas complejas de los patrones detectados en la capa flatten.
   - Usa también `ReLU` como función de activación.

4. **Capa Densa**
   - Capa de salida del modelo.
   - Devuelve un único valor como resultado final.
   - Se puede usar para regresión o clasificacion binaria.

### Justificacion del diseño

- Ideal para tareas de predicción en series de tiempo pequeñas, como:
  - Predicción de fallas
  - Detección de anomalias
  - Clasificación basada en eventos breves
