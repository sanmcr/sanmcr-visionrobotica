---
layout: post
title: Follow Line
---

# Reporte Técnico Detallado: Implementación de Seguimiento de Línea con PID en Robótica

## 1. Introducción
Este proyecto tiene como objetivo la implementación de un sistema de seguimiento de línea utilizando un robot autónomo controlado por visión computacional y un controlador PID (Proporcional-Integrativo-Derivativo). La idea es permitir que el robot siga una línea roja sobre un fondo blanco, ajustando su dirección y velocidad basándose en el análisis de imágenes en tiempo real. La detección de la línea se realiza mediante el uso de filtros y técnicas avanzadas de procesamiento de imágenes, y el comportamiento del robot se ajusta utilizando un controlador PID para mantener el seguimiento eficiente y estable.
Durante el proceso, se presentaron varios desafíos, como la detección precisa de la línea en diferentes condiciones de iluminación y la gestión de los giros del robot. Este informe explica en detalle cómo se abordaron estos problemas, las técnicas utilizadas y los resultados obtenidos.





## 2. Descripción del Proceso y Soluciones Técnicas Implementadas

### 2.1. Adquisición y Preprocesamiento de Imágenes


#### Adquisición de la Imagen

El sistema utiliza una cámara incorporada en el robot para capturar imágenes del entorno en tiempo real. Estas imágenes sirven como la entrada al sistema de visión por computadora y son procesadas a una alta frecuencia para permitir el seguimiento continuo de la línea.

#### Conversión al Espacio de Color HSV
Uno de los primeros pasos en el procesamiento de las imágenes es la conversión de la imagen del espacio de color BGR (azul, verde, rojo) a HSV (matiz, saturación, valor). El espacio HSV es particularmente útil en visión por computadora porque separa el componente de color (matiz) de los componentes de brillo (saturación y valor), lo que facilita la detección de colores específicos en condiciones de iluminación cambiantes.



```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

#### Definición de los Rangos de Color Rojo
Para detectar la línea roja, se definieron dos rangos en el espacio de color HSV. Esto se debe a que el color rojo tiene dos tonalidades: una cerca del valor 0 (rojo claro) y otra cerca del valor 180 (rojo oscuro) en el espectro HSV.

    • Rango inferior del rojo claro: lower_red1 = np.array([0, 120, 70])
    • Rango superior del rojo claro: upper_red1 = np.array([10, 255, 255])
    • Rango inferior del rojo oscuro: lower_red2 = np.array([170, 120, 70])
    • Rango superior del rojo oscuro: upper_red2 = np.array([180, 255, 255])

Estas máscaras se crean con la función cv2.inRange, que devuelve una imagen binaria donde los píxeles dentro del rango especificado son blancos (255) y los que están fuera del rango son negros (0).


```python
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)
```

#### Filtrado Morfológico

Una vez que se ha generado la máscara de la línea roja, se aplica un filtro morfológico para reducir el ruido. Los filtros morfológicos como la operación de apertura y cierre se utilizan para limpiar la máscara y eliminar pequeñas manchas que no forman parte de la línea.


```python
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

### 2.2 Detección de Contornos

Una vez que se ha limpiado la máscara de la línea roja, el siguiente paso es detectar los contornos de la línea utilizando la función cv2.findContours. Este paso es crucial porque nos permite identificar la forma de la línea y su ubicación en la imagen.


```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

Aquí se utiliza `cv2.RETR_EXTERNAL` para detectar solo los contornos externos, ya que la línea roja es el contorno principal que queremos rastrear.

### 2.3. Cálculo del Centro de la Línea

Una vez que se han detectado los contornos, el siguiente paso es encontrar el centro de la línea en la imagen. Esto se hace calculando el centroide del contorno de la línea utilizando los momentos de la imagen. Los momentos proporcionan la información necesaria para calcular el centro de masa del contorno.


```python
M = cv2.moments(largest_contour)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
```

El cálculo del centro de la línea `(cX, cY)` es crucial para determinar la dirección en la que el robot debe moverse.


### 2.4. Control PID para Seguimiento de la Línea

#### Proceso de Control PID

El controlador PID ajusta la dirección y la velocidad del robot en función del error calculado entre el centro de la imagen y el centro de la línea. El error se calcula como la diferencia entre el centro de la imagen y la posición de la línea detectada en el eje X.

```python
error = cX - frame_center
```

- PID - **Proporcional (P)**: Este término calcula la cantidad de corrección basada directamente en el error.

    ```python
    p_control = Kp * error
    ```
- PID - **Derivativo (D)**: Este término calcula la tasa de cambio del error, lo que ayuda a reducir las oscilaciones y mejora la estabilidad.

    ```python
    d_control = Kd * (error - prev_error)
    ```
- PID - **Integral (I)**: Este término acumula el error a lo largo del tiempo. Inicialmente, se probó con la ganancia integral activada, pero se encontró que acumulaba errores y provocaba inestabilidad, por lo que se decidió desactivarla en la mayoría de las pruebas.
    ```python
    integral += error
    i_control = Ki * integral
    ```

El control PID se usa para calcular el valor de la dirección (steering):

```python
raw_steering = p_control + i_control + d_control
```

#### Suavizado de la Dirección

Una de las soluciones implementadas para evitar las oscilaciones excesivas fue el uso de un filtro exponencial para suavizar los cambios en la dirección del robot. Este filtro suaviza las transiciones de la dirección, haciendo que el robot no reaccione de forma demasiado brusca a los pequeños errores.


```python
steering = alpha * prev_steering + (1 - alpha) * raw_steering
prev_steering = steering
```

El valor de alpha se ajustó para obtener una respuesta más suave. Un valor de alpha = 0.3 proporcionó un buen equilibrio entre velocidad de respuesta y suavidad.

#### Ajuste Dinámico de Velocidad

Se implementó un ajuste dinámico de la velocidad en función de la curvatura de la línea. En curvas cerradas, la velocidad se reduce para evitar que el robot se desestabilice.

```python
if abs(error) < 30:
    velocity = 5.0  # Recta
elif abs(error) >= 30 and abs(error) < 50:
    velocity = 3.0  # Curva moderada
else:
    velocity = 2.0  # Curva cerrada
```















