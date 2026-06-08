import { motion } from "framer-motion";
import { title as content } from "../content";
import { RichText } from "../components/RichText";

export function TitleSlide() {
  return (
    <div
      className="relative flex h-full w-full flex-col overflow-hidden"
      style={{ background: "var(--color-uni-yellow)" }}
    >
      {/* Mosaico geométrico negro+blanco en la esquina superior derecha */}
      <Mosaic className="absolute right-[5%] top-[8%] z-0 h-[140px]" />

      {/* Red neuronal estilizada como acento temático, banda central derecha */}
      <NeuralNetDiagram className="title-neural-net absolute right-[3%] top-[28%] z-0 h-[38%] w-[34%]" />

      {/* Banda negra inferior a sangre como cierre */}
      <div
        aria-hidden
        className="absolute inset-x-0 bottom-0 z-0 h-[3.5%]"
        style={{ background: "var(--color-uni-black)" }}
      />

      {/* Contenido en columna interior */}
      <div className="relative z-10 flex h-full flex-col px-[8%] pt-[4%] pb-[5%]">
        {/* Cabecera: monograma .h + nombre institucional */}
        <motion.div
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="flex flex-col"
        >
          <img
            src="/logo-uni.png"
            alt="Universidad de las Hespérides"
            className="h-40 w-auto self-start max-w-none"
          />
        </motion.div>

        {/* Título */}
        <div className="mt-[4%] flex-1">
          <motion.h1
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
            className="title-heading max-w-[60%] font-display text-[clamp(2.4rem,5.6vw,5.2rem)] font-medium leading-[1.04] tracking-tight text-[var(--color-uni-black)]"
          >
            {content.title.lineOne}
            <br />
            <RichText fragments={content.title.lineTwo} />
          </motion.h1>

          <motion.div
            initial={{ opacity: 0, scaleX: 0 }}
            animate={{ opacity: 1, scaleX: 1 }}
            transition={{ delay: 0.55, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="mt-7 h-[3px] w-24 origin-left"
            style={{ background: "var(--color-uni-black)" }}
          />

          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.55, duration: 0.85 }}
            className="font-display mt-6 text-[clamp(1.1rem,1.9vw,1.7rem)] italic text-[var(--color-uni-black)]/80"
          >
            {content.subtitle}
          </motion.p>
        </div>

        {/* Pie: autores a la izquierda, asignatura+curso a la derecha */}
        <div className="mt-auto flex items-end justify-between gap-8">
          {/* Tarjeta de autores: cabecera negra + cuerpo crema */}
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.85, duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
            className="w-full max-w-[28rem] overflow-hidden rounded-2xl shadow-[0_8px_30px_rgba(0,0,0,0.12)]"
          >
            <div
              className="px-5 py-2.5 text-xs font-semibold uppercase tracking-[0.18em] text-white"
              style={{ background: "var(--color-uni-black)" }}
            >
              {content.authorsLabel}
            </div>
            <div
              className="px-5 py-4 text-base leading-7 text-[var(--color-uni-black)]"
              style={{ background: "#fffaf0" }}
            >
              {content.authors.map((author) => (
                <div key={author}>{author}</div>
              ))}
            </div>
          </motion.div>

          {/* Asignatura + curso */}
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.95, duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
            className="flex flex-col items-end text-right"
          >
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-[var(--color-uni-black)]/85">
              <span
                className="inline-block h-[6px] w-[6px]"
                style={{ background: "var(--color-uni-black)" }}
              />
              {content.asignaturaLabel}
            </div>
            <div className="font-display mt-2 text-lg leading-snug text-[var(--color-uni-black)] max-w-[22rem]">
              {content.subject}
            </div>
            <div className="font-display mt-1 text-base italic text-[var(--color-uni-black)]/75">
              {content.course}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}

function Mosaic({ className = "" }: { className?: string }) {
  // 2 columnas × 3 filas alternando negro / fondo (crema sobre amarillo).
  const CREAM = "#fffaf0";
  const cells = [
    [0, 0, "black"],
    [1, 0, "cream"],
    [0, 1, "cream"],
    [1, 1, "black"],
    [0, 2, "black"],
    [1, 2, "cream"],
  ] as const;
  return (
    <motion.svg
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15, duration: 0.7 }}
      viewBox="0 0 56 84"
      className={className}
      preserveAspectRatio="xMidYMid meet"
    >
      {cells.map(([cx, cy, color], i) => (
        <rect
          key={i}
          x={cx * 30}
          y={cy * 30}
          width={26}
          height={26}
          fill={color === "cream" ? CREAM : "var(--color-uni-black)"}
        />
      ))}
    </motion.svg>
  );
}

function NeuralNetDiagram({ className = "" }: { className?: string }) {
  // Capas: 3 inputs, 3 capas ocultas de 5 nodos, 1 output.
  // Dimensiones internas en una caja 200x130, con el SVG escalando.
  const layers: { x: number; ys: number[] }[] = [
    { x: 18, ys: [30, 65, 100] },
    { x: 65, ys: [22, 45, 65, 85, 108] },
    { x: 110, ys: [22, 45, 65, 85, 108] },
    { x: 155, ys: [22, 45, 65, 85, 108] },
    { x: 195, ys: [65] },
  ];

  // Generamos las conexiones entre capas consecutivas.
  const edges: { x1: number; y1: number; x2: number; y2: number }[] = [];
  for (let l = 0; l < layers.length - 1; l++) {
    const a = layers[l];
    const b = layers[l + 1];
    for (const ya of a.ys) {
      for (const yb of b.ys) {
        edges.push({ x1: a.x, y1: ya, x2: b.x, y2: yb });
      }
    }
  }

  return (
    <motion.svg
      aria-hidden
      className={className}
      viewBox="0 0 215 145"
      preserveAspectRatio="xMidYMid meet"
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: 0.2, duration: 1, ease: [0.22, 1, 0.36, 1] }}
    >
      {/* Conexiones */}
      {edges.map((e, i) => (
        <line
          key={i}
          x1={e.x1}
          y1={e.y1}
          x2={e.x2}
          y2={e.y2}
          stroke="var(--color-uni-black)"
          strokeOpacity="0.22"
          strokeWidth="0.4"
        />
      ))}

      {/* Nodos */}
      {layers.map((layer, l) =>
        layer.ys.map((y, i) => {
          const isInput = l === 0;
          const isOutput = l === layers.length - 1;
          const r = isInput ? 4 : isOutput ? 5 : 3.2;
          const fill = isOutput
            ? "var(--color-uni-black)"
            : isInput
            ? "#fffaf0"
            : "#fffaf0";
          return (
            <circle
              key={`${l}-${i}`}
              cx={layer.x}
              cy={y}
              r={r}
              fill={fill}
              stroke="var(--color-uni-black)"
              strokeWidth={isOutput ? 0 : 0.9}
            />
          );
        }),
      )}

      {/* Etiquetas θ -> P */}
      <text
        x={18}
        y={132}
        textAnchor="middle"
        fontFamily="Fraunces, serif"
        fontStyle="italic"
        fontSize="9"
        fill="var(--color-uni-black)"
      >
        θ
      </text>
      <text
        x={195}
        y={132}
        textAnchor="middle"
        fontFamily="Fraunces, serif"
        fontStyle="italic"
        fontSize="9"
        fill="var(--color-uni-black)"
      >
        P
      </text>
      <line
        x1={28}
        y1={129}
        x2={185}
        y2={129}
        stroke="var(--color-uni-black)"
        strokeOpacity="0.55"
        strokeWidth="0.5"
      />
      <polygon
        points="185,127 190,129 185,131"
        fill="var(--color-uni-black)"
        fillOpacity="0.7"
      />
    </motion.svg>
  );
}
