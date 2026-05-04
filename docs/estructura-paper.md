# Estructura del paper

Este documento propone la estructura que deberíamos seguir al redactar la memoria final. La idea es que cada apartado tenga un papel claro y que el conjunto cuente una historia coherente: primero situamos el problema, luego explicamos las herramientas que vamos a usar, después describimos lo que hacemos, presentamos los resultados con sus matices, y cerramos con una reflexión sobre lo aprendido y hacia dónde podría continuar el trabajo. La extensión total razonable, contando todo, estaría entre treinta y cincuenta páginas.

## 1. Portada y resumen

La portada incluye el título del trabajo, los autores, la asignatura, el tutor y la fecha. Después va un resumen breve, de unas diez o quince líneas, que explica en lenguaje claro qué hemos hecho, por qué, y cuáles son los resultados principales. El resumen es lo primero que va a leer un lector apurado, así que tiene que ser autocontenido y dejar claro que el trabajo va sobre el uso de redes neuronales como aproximadores numéricos de modelos de pricing de opciones, no sobre predicción de precios de mercado.

## 2. Introducción

La introducción es donde justificamos por qué este trabajo merece la pena. Empieza con una motivación general sobre el problema de evaluar modelos de pricing complejos, sigue con la idea de los deep surrogates como respuesta a ese problema, plantea explícitamente la pregunta de investigación que ya tenemos formulada en la planificación, enumera los objetivos concretos, lista las contribuciones que aporta el trabajo y termina con un mapa breve de cómo está organizado el resto del documento. La introducción no debería ocupar más de tres o cuatro páginas y tiene que poder leerse de forma independiente del resto.

## 3. Marco teórico

Aquí explicamos las herramientas matemáticas y financieras que vamos a usar, antes de meternos con la metodología propia. Conviene dividirlo en tres bloques. El primer bloque cubre los modelos de pricing de opciones que vamos a usar como referencia: Black-Scholes con su fórmula cerrada y la deducción intuitiva de las Deltas analíticas, y Heston con su dinámica de volatilidad estocástica y la solución mediante Fourier. El segundo bloque introduce los conceptos financieros que necesitamos para evaluar el surrogate: volatilidad implícita, moneyness, Greeks, y por qué la volatilidad implícita es una métrica más informativa que el precio absoluto. El tercer bloque es sobre las redes neuronales como aproximadores funcionales: arquitectura básica de un MLP, funciones de activación, teorema de aproximación universal, diferenciación automática y por qué las redes son adecuadas en alta dimensión frente a polinomios, splines, sparse grids o Gaussian processes.

## 4. Estado del arte y revisión bibliográfica

Esta sección recoge todo el recorrido bibliográfico que hemos hecho y que ya está descrito en la planificación. Empieza con el trabajo histórico de Hutchinson, Lo y Poggio como punto de partida del uso de redes en pricing, sigue con la revisión de Ruf y Wang como mapa general del campo, menciona la línea de optimal stopping de Becker, Cheridito y Jentzen para diferenciar nuestro enfoque, dedica el espacio principal al paper de Chen, Didisheim y Scheidegger como referencia central, y cierra con Huge y Savine sobre differential machine learning, que conecta con uno de nuestros experimentos opcionales. Después una vista de pájaro de los papers complementarios que tenemos en `papers/others/` para situar nuestro trabajo dentro de un panorama más amplio. La sección termina justificando por qué el enfoque de deep surrogates es el que mejor encaja con un trabajo de métodos numéricos.

## 5. Metodología

Aquí describimos cómo hemos diseñado el surrogate, sin entrar todavía en los resultados. La metodología tiene varias piezas que conviene explicar por separado pero como un flujo continuo. Primero, la generación de datos sintéticos: cómo definimos el hipercubo de inputs, qué distribución de muestreo usamos, cómo evaluamos el modelo verdadero para producir las etiquetas y cómo dividimos el dataset en entrenamiento, validación y test. Segundo, la arquitectura del surrogate: número de capas, anchura, función de activación, normalización de inputs y output. Tercero, la lógica de entrenamiento: función de pérdida, optimizador, learning rate, batch size, criterio de parada. Cuarto, las métricas de evaluación: MAE en precio, MAE en volatilidad implícita, MAE en Delta, distribución del error por regiones de moneyness y vencimiento, y comparación de tiempos frente al solver original. Esta sección debe escribirse con suficiente detalle para que alguien pudiera reproducir los experimentos.

## 6. Experimentos y resultados

Es el núcleo del trabajo y la sección más extensa. La organizamos siguiendo el orden lógico de los experimentos. Empieza con el surrogate de Black-Scholes como caso de validación, donde comparamos precios y Deltas con la solución analítica y mostramos que el pipeline funciona. Sigue con la extensión a Heston, que es donde la metodología empieza a tener valor real porque el solver original ya es costoso. Continúa con el análisis del error por regiones de moneyness y vencimiento mediante mapas de calor o tablas, para mostrar dónde el surrogate funciona mejor y dónde se deteriora. Después viene la comparación de funciones de activación, mostrando el resultado clave de que dos redes con MAE de precio similar pueden tener Deltas muy distintas. Luego la comparación entre muestreo uniforme y muestreo enfocado, donde discutimos si concentrar puntos en regiones difíciles mejora el error en esas zonas. Sigue la sección de eficiencia computacional, comparando tiempos de evaluación entre el solver original y el surrogate. Finalmente, si llegamos, el experimento opcional sobre función de pérdida y differential machine learning, donde comparamos L1, L2 y Huber, y probamos a entrenar la red usando también las derivadas verdaderas. Cada experimento debe presentarse con la pregunta concreta que responde, los detalles del setup, los resultados visualizados de forma clara y una interpretación corta.

## 7. Discusión

La discusión recoge las conclusiones transversales que se sacan al juntar todos los experimentos. Aquí es donde conectamos los resultados particulares con la pregunta de investigación principal y donde discutimos los matices que han ido apareciendo. Por ejemplo, hasta qué punto el surrogate generaliza bien fuera de la zona donde se ha entrenado, qué papel juega la suavidad de la red en la calidad de los Greeks, en qué condiciones compensa el coste inicial de generar datos y entrenar la red, y cuáles son las limitaciones que hemos encontrado. La discusión también es el lugar adecuado para hablar de modelos más ambiciosos como Stochastic Local Volatility, rough Bergomi o Bates con saltos, explicando que la metodología que hemos desarrollado se diseña pensando en ellos aunque no los hayamos implementado.

## 8. Conclusiones y líneas futuras

Las conclusiones son una recapitulación breve de lo que hemos hecho, qué hemos aprendido y qué responde el trabajo a la pregunta de investigación. No es una repetición del resumen, sino una mirada retrospectiva con perspectiva. Después abrimos las líneas futuras: aplicar la metodología a modelos más complejos, profundizar en differential ML, explorar muestreo adaptativo de verdad con active learning, o conectar el surrogate con un problema de calibración real. Esta sección debe ser corta y honesta sobre los límites del alcance del trabajo.

## 9. Bibliografía

La bibliografía recoge todos los papers, libros y recursos que hemos citado en el documento. Conviene gestionarla con BibTeX desde el principio para no tener que arreglar el formato a última hora. Como mínimo deben aparecer los cinco papers del núcleo de la revisión bibliográfica y los papers de la carpeta `papers/others/` que hayamos citado de pasada.

## 10. Anexos

Los anexos son para material que es útil pero que rompería el flujo si lo metemos en el cuerpo principal. Algunas cosas que tendría sentido poner aquí son detalles matemáticos extensos como la derivación completa de la fórmula de Heston por Fourier, tablas de hiperparámetros completas, gráficos secundarios, código relevante o instrucciones de reproducibilidad. La regla es que el cuerpo principal tiene que ser autocontenido, los anexos son material de apoyo para quien quiera profundizar.

## Cómo encarar la redacción

Una recomendación práctica: no escribir las secciones en orden de aparición. Lo más eficiente es escribir primero la metodología y los experimentos, porque son las partes que reflejan directamente lo que hemos hecho y que evolucionan menos a posteriori. Después se escribe el marco teórico y la revisión bibliográfica con la perspectiva de lo que se ha necesitado realmente. La introducción y el resumen se escriben al final, cuando ya sabemos exactamente qué historia estamos contando. Las conclusiones y la discusión también van al final por la misma razón.
