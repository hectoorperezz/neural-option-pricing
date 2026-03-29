# Redes neuronales como aproximadores para pricing

Usar redes neuronales para aprender la función de pricing $f(\theta) \to P$ directamente, evitando el coste de resolver el modelo (MC o PDE) cada vez. Investigar si una red entrenada offline puede reemplazar al solver numérico en tiempo real.

## Cuestiones a investigar

- ¿Cómo se plantea el problema de pricing como aprendizaje supervisado?
- ¿Qué arquitectura usar? ¿Cuántos datos de entrenamiento se necesitan?
- Entrenar una red para aproximar Black–Scholes (validación) y luego para Heston (donde el solver es caro).
- Medir error de aproximación, velocidad de inferencia y capacidad de generalización fuera de la muestra de entrenamiento.

## Referencias

- Hutchinson, Lo y Poggio (1994), *"A nonparametric approach to pricing and hedging derivative securities"*.
- Bayer et al. (2019), *"Deep optimal stopping"*.
- Ruf y Wang (2020), *"Neural networks for option pricing and hedging"*.

Los PDFs de estas tres referencias están en `papers/`. En `papers/others/` hay papers adicionales de interés sobre el tema.
