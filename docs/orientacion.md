# Orientación del Proyecto Final  
## Neural Networks as Surrogate Solvers for Option Pricing and Calibration

La idea es:

```text
Modelo financiero + solver numérico
                ↓
        Dataset sintético de precios
                ↓
        Entrenamiento de la NN
                ↓
        NN surrogate para pricing rápido y calibración
```

Este enfoque encaja mejor con Métodos Numéricos porque el foco está en el trade-off entre precisión y coste computacional. Además, los costes computacionales de calibración e inferencia son temas muy de actualidad para el sector.


Opción de título:

**Accelerating Option Pricing and Calibration with Neural Network Surrogate Solvers**

Tésis que proponemos:

> Los métodos clásicos de pricing y calibración pueden ser costosos cuando deben evaluarse muchas veces. Una red neuronal puede entrenarse offline con precios generados por un solver numérico y usarse después online como aproximador rápido, reduciendo tiempos de ejecución con un error controlado.


## Enfoque

El core debe ser:

```text
NN as surrogate pricing solver
```

La red aprende el mapa forward:

`f_φ(m, T, Θ) ≈ C_solver / K`

donde `m = S0/K`, `T` es el vencimiento, `Θ` son los parámetros del modelo y `C_solver` es el precio calculado por el solver clásico.

El enfoque alternativo sería que la red aprendiera directamente:

```text
superficie de precios/volatilidades → Θ
```

Esto sería una **direct calibration map**, pero es más complejo y menos controlable. Conviene dejarlo como comparación conceptual o posible extensión, no como núcleo.


El proyecto debería tener tres bloques principales.

Primero, construir un **surrogate solver para pricing**. Se puede empezar con Black-Scholes, aunque no sea caro, se puede usar este paso para validar la idea de fondo:

`f_φ(m, T, r, σ) ≈ C_BS / K`

Después, pasamos a un solver más interesante, por ejemplo Heston, porque es costoso y conecta muy bien con la calibración.

Segundo, hacer una **calibración sintética acelerada**. La calibración clásica busca parámetros que reproduzcan precios objetivo:

```text
Θ̂ = argmin_Θ  Σ_i  ( C_solver(K_i, T_i; Θ) − C_i^target )²
```

Con la red, sustituimos el solver por el surrogate:

```text
Θ̂_NN = argmin_Θ  Σ_i  ( f_φ̂(K_i, T_i, Θ) − C_i^target )²
```


En tercer lugar, comparar **error y tiempo**. El objetivo es medir cuánto error introduce la red y cuánto tiempo ahorra:

`Speed-up = t_solver / t_NN`

También se deben comparar los parámetros recuperados en calibración sintética:

`‖Θ̂_solver − Θ*‖`

`‖Θ̂_NN − Θ*‖`

## Ejemplo con Heston

Heston es un buen candidato porque tiene varios parámetros y se calibra habitualmente. Sus parámetros principales son:

`Θ = (v₀, θ, κ, ξ, ρ)`

donde `v0` es la varianza inicial, `θ` la varianza media de largo plazo, `κ` la velocidad de reversión a la media, `ξ` la volatilidad de la volatilidad y `ρ` la correlación entre precio y varianza.

La red aprendería:

`f_φ(m, T, v₀, θ, κ, ξ, ρ) ≈ C_Heston / K`

Después se eligen unos parámetros verdaderos `Θ*`, se generan precios sintéticos con el solver y se intenta recuperar `Θ*` calibrando con el solver clásico y con la red. Aquí lo interesante es comparar precisión y coste computacional.


## Parte opcional: optimal stopping

Como parte opcional podemos incluir una investigación adicional, que conecte con opciones American/Bermudan y el paper de Becker et al. En este caso, la red no solo aproxima precios, sino continuation values o reglas de ejercicio. Esto es lo que se conoce como el problema de optimal stopping.

El problema general es:

```text
V₀ = sup_{τ ∈ 𝒯}  E[ exp(−r·τ) · g(S_τ) ]
```

## Posible Estructura para el paper final

```text
1. Introducción
   Motivación: coste computacional en pricing y calibración.

2. Revisión de literatura
   Diferencia entre redes con datos de mercado y redes como surrogate solvers.
   Ruf & Wang para contexto.
   Becker et al. para optimal stopping (si finalmente se incluye).

3. Metodología
   Modelo elegido.
   Solver clásico.
   Dataset sintético.
   Arquitectura NN.
   Métricas de error y tiempo.

4. Experimentos
   Black-Scholes.
   Surrogate de Heston.
   Calibración sintética.
   Comparación error/tiempo.

5. Extensión opcional
   Optimal stopping.
   American/Bermudan options.
   Becker et al.

6. Discusión
   Ventajas.
   Limitaciones.
   Extrapolación.
   Greeks.
   Restricciones de no arbitraje.

7. Conclusión
   Cuándo merece la pena usar NN como surrogate solver.
```
