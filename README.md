# Surrogates neuronales para el pricing de opciones: precisión, volatilidad implícita y calibración bajo Heston

> Proyecto final de **Métodos Numéricos y Simulación Estocástica**
> Máster en Finanzas Cuantitativas y Métodos Computacionales · **Universidad de las Hespérides** · Curso 2025/2026
> Ángel Fernández Sánchez · Jorge Alfageme Sotillos · Héctor Pérez Ledesma

Una red neuronal puede aprender la **función de pricing** de un modelo —el mapa de parámetros a precio— e imitar al solver que la genera. Entrenada *offline* con datos sintéticos, sustituye al solver justo donde hace falta valorar millones de veces (calibración diaria, Greeks, Montecarlo), siendo órdenes de magnitud más rápida y, por construcción, diferenciable.

Usamos **Black-Scholes como banco de validación** —tiene solución cerrada, así que podemos medir el error exacto en precio, Delta y volatilidad implícita— y **Heston como caso principal**, donde el precio es una integral de Fourier costosa y el surrogate empieza a tener sentido práctico.

> **Pregunta de investigación.** ¿Bajo qué condiciones una red profunda actúa como un surrogate preciso, rápido y diferenciable de una función de pricing?

El estudio se articula en **seis experimentos** sobre **catorce surrogates**, cada uno aislando una variable: métrica de evaluación, activación, distribución del muestreo, eficiencia computacional, información diferencial en la pérdida, y profundidad junto al scheduler de *learning rate*.


## Estructura

| Carpeta | Contenido |
|---|---|
| `src/` | Solvers (Black-Scholes, Heston), generación de datos, modelos, entrenamiento y evaluación por bins |
| `scripts/` | Pipelines que reproducen tablas y figuras |
| `docs/` | Paper en LaTeX, metodología y notas de los experimentos |
| `playground/` | Laboratorio interactivo (FastAPI + React) para probar los surrogates reales |
| `papers/` | Bibliografía consultada |

## Para empezar

- El **paper** completo está en `docs/latex/` (PDF).
- El **playground** (`playground/`) es un laboratorio interactivo para valorar, ver dónde falla la extrapolación y calibrar con los surrogates reales.
- La **bibliografía completa** está en el paper; los PDFs de las referencias núcleo están en `papers/`.
