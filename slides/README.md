# Slides

Presentación del proyecto **«Redes neuronales como aproximadores numéricos para
pricing de opciones bajo Black-Scholes y Heston»** construida como una
*single-page app* en React. Sustituye al PowerPoint tradicional y nos da
control total sobre transiciones, tipografía y motivos visuales que ya usa el
paper en LaTeX.

## Arranque rápido

Desde la raíz del repositorio:

```bash
cd slides
npm install     # solo la primera vez
npm run dev
```

Vite abre el navegador en `http://localhost:5173`. Si no se abre solo,
copia esa URL en el navegador. Atajos: `→` siguiente, `←` anterior,
`Espacio` siguiente, `Home`/`End` ir a primera/última, `1..9` salto directo.

Otros comandos útiles:

| Comando             | Qué hace                                  |
| ------------------- | ----------------------------------------- |
| `npm run dev`       | Servidor de desarrollo con *hot reload*.  |
| `npm run build`     | Compila la versión de producción a `dist/`. |
| `npm run preview`   | Sirve la versión compilada para revisar.  |

## Pila

- **Vite + React 19 + TypeScript**.
- **Tailwind CSS v4** (configuración vía `@theme` en `src/index.css`, sin
  `tailwind.config.js`).
- **Framer Motion** para transiciones entre diapositivas y animaciones de
  entrada por bloques.
- Tipografías servidas desde Google Fonts: `Fraunces` (display, serif
  moderno) e `Inter` (texto) con `JetBrains Mono` para tokens técnicos.

## Atajos de teclado (resumen completo)

| Tecla(s)                         | Acción              |
| -------------------------------- | ------------------- |
| `→` · `↓` · `Espacio` · `PgDn`   | Siguiente diapositiva |
| `←` · `↑` · `PgUp`              | Diapositiva anterior |
| `Home` · `End`                   | Primera / última    |
| `1` … `9`                        | Salto directo       |

La diapositiva actual también se refleja en el *hash* de la URL (`#1`, `#2`,
…), así que puedes compartir un enlace que abra la presentación en una
diapositiva concreta.

## Estructura

```
slides/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig*.json
├── public/
│   ├── logo.png        Logo institucional completo
│   └── h-mono.png      Monograma .h
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── index.css       Paleta + utilidades Tailwind (@theme)
    ├── hooks/
    │   └── useSlideNavigation.ts
    ├── components/
    │   ├── SlideStage.tsx     AnimatePresence + transición horizontal
    │   ├── SlideChrome.tsx    Cromo inferior con número y botones
    │   ├── ProgressBar.tsx    Barra superior de progreso
    │   └── SlideLayout.tsx    Eyebrow, Headline, Body, Stagger, FooterStamp
    └── slides/
        ├── index.tsx          Registro central (orden y composición)
        ├── TitleSlide.tsx
        ├── QuestionSlide.tsx
        ├── MotivationSlide.tsx
        ├── SurrogateSlide.tsx
        ├── CasesSlide.tsx
        ├── ExperimentsOverviewSlide.tsx
        ├── ExperimentSlide.tsx        Componente parametrizado E1..E5
        └── ClosingSlide.tsx
```

## Editar el texto de las diapositivas

Todo el contenido textual está centralizado en `src/content/`, **un archivo
JSON por diapositiva** (más uno consolidado para los cinco experimentos).
Si solo quieres cambiar texto, ahí es donde tocas; no hace falta tocar
ningún `.tsx`.

```
src/content/
├── title.json                  Portada
├── question.json               Pregunta de investigación
├── motivation.json             Motivación + tarjetas
├── surrogate.json              Pipeline del surrogate
├── cases.json                  Black-Scholes vs Heston
├── experiments-overview.json   Cuadrícula de los cinco experimentos
├── experiments.json            Array con E1..E5
├── closing.json                Cierre
├── types.ts                    Tipos TS de cada JSON
└── index.ts                    Punto de entrada tipado
```

Convenciones útiles dentro de los JSON:

- **Texto plano** se escribe directamente: `"body": "..."`.
- **Texto con énfasis** se escribe como array de fragmentos:
  ```json
  "headline": [
    "Cuanto más realista es el modelo, ",
    { "text": "más cara es cada evaluación.", "highlight": true }
  ]
  ```
  Cada fragmento puede llevar `highlight: true` (subrayado amarillo),
  `italic: true` o `math: true` (renderiza el `text` como LaTeX).
- **Fórmulas matemáticas inline** dentro de un string se delimitan con
  `$...$`, igual que LaTeX:
  ```json
  "bullets": ["$\\Delta = e^{-qT} P_1$; con $q = 0$, $\\Delta = P_1$."]
  ```
- Las **barras invertidas de LaTeX deben escaparse** dentro del JSON:
  `\Delta` se escribe `\\Delta`.

Después de editar, Vite recarga automáticamente. Si TypeScript te grita,
revisa `src/content/types.ts`: es la fuente de verdad del esquema.

## Añadir una diapositiva nueva

1. Crea un componente en `src/slides/MiSlide.tsx` que renderice su contenido.
   Usa los primitivos de `components/SlideLayout` (`Eyebrow`, `Headline`,
   `Body`, `Stagger`, `FooterStamp`) para mantener consistencia visual.
2. Impórtalo y añádelo a la lista `slides` en `src/slides/index.tsx`, en la
   posición que toque del guión.
3. Si es una nueva variante recurrente (otro patrón de slide), considera
   añadirla a `components/SlideLayout` para que sea reutilizable.

## Añadir resultados a los slides de experimento

Los slides E1..E5 están parametrizados con `ExperimentSlide`. Cuando un
experimento tenga resultados, basta con pasar la prop `results` con el JSX
correspondiente (tabla, figura, párrafo de conclusión):

```tsx
<ExperimentSlide
  id="E2"
  title="..."
  question="..."
  metric="..."
  surrogates={["..."]}
  results={<MyE2Results />}
/>
```

Mientras `results` no se especifique, se muestra un placeholder discreto.

## Paleta institucional

| Token CSS                       | Valor      | Uso |
| ------------------------------- | ---------- | --- |
| `--color-uni-yellow`            | `#FFD100`  | Acento principal Hespérides |
| `--color-uni-yellow-soft`       | `#FDEA99`  | Fondos crema suaves |
| `--color-uni-cream`             | `#FFF8DC`  | Fondos amarillo crema |
| `--color-uni-black`             | `#0E0E0E`  | Tinta institucional |
| `--color-ink` / `-soft`/ `-muted` | grises  | Jerarquía de texto |

Los acentos secundarios (`accent-red`, `accent-green`, `accent-orange`,
`accent-purple`) están reservados para gráficos cuantitativos, en coherencia
con la paleta del paper.
