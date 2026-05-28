import { motion } from "framer-motion";

interface SlideBackdropProps {
  /** Si es portada o cierre se ocultan los elementos para dejar el lienzo libre. */
  bleed?: boolean;
}

/**
 * Capa decorativa de fondo común a todas las slides interiores: banda
 * amarilla a sangre por debajo, halftone tenue en la esquina superior
 * derecha y un fondo crema muy sutil. Sin pesar sobre el contenido.
 */
export function SlideBackdrop({ bleed = false }: SlideBackdropProps) {
  if (bleed) return null;

  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 z-0">
      {/* Banda inferior fina a sangre */}
      <div
        className="absolute inset-x-0 bottom-0 h-[10px]"
        style={{ background: "var(--color-uni-yellow)" }}
      />

      {/* Halftone amarillo tenue arriba a la derecha */}
      <motion.svg
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.9 }}
        className="absolute right-0 top-0 h-44 w-44"
        viewBox="0 0 100 100"
        preserveAspectRatio="xMaxYMin meet"
      >
        {Array.from({ length: 8 }).map((_, i) =>
          Array.from({ length: 6 }).map((_, j) => {
            const cx = 100 - i * 11;
            const cy = j * 11 + 8;
            const dist = Math.hypot(i, j);
            const r = Math.max(0, 4.2 - dist * 0.45);
            if (r <= 0.1) return null;
            return (
              <circle
                key={`${i}-${j}`}
                cx={cx}
                cy={cy}
                r={r}
                fill="var(--color-uni-yellow)"
              />
            );
          }),
        )}
      </motion.svg>
    </div>
  );
}
