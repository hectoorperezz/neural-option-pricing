import { motion } from "framer-motion";

interface SideRailProps {
  index: number;
  total: number;
}

/**
 * Carril lateral izquierdo que aporta identidad institucional a todas las
 * slides interiores. Combina:
 *  - una banda vertical amarilla fina a sangre,
 *  - el número de slide grande en serif sobre crema,
 *  - una marca con el monograma .h al pie.
 */
export function SideRail({ index, total }: SideRailProps) {
  const display = String(index + 1).padStart(2, "0");

  return (
    <aside
      aria-hidden
      className="pointer-events-none absolute inset-y-0 left-0 z-10 hidden w-[5rem] flex-col items-center justify-between py-14 pl-[14px] sm:flex"
    >
      {/* Banda vertical amarilla a sangre */}
      <div
        className="absolute inset-y-0 left-0 w-[6px]"
        style={{ background: "var(--color-uni-yellow)" }}
      />

      {/* Marca .h arriba */}
      <motion.img
        src="/h-mono.png"
        alt=""
        className="h-6 w-auto opacity-80"
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 0.8, y: 0 }}
        transition={{ duration: 0.6 }}
      />

      {/* Número grande, vertical */}
      <motion.div
        key={index}
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="font-display flex flex-col items-center leading-none"
      >
        <span className="text-[clamp(2.4rem,3.6vw,3.4rem)] font-medium text-[var(--color-uni-black)]">
          {display}
        </span>
        <span className="mt-1 h-px w-6 bg-[var(--color-uni-black)]/30" />
        <span className="mt-1 font-mono text-[10px] tracking-[0.25em] text-[var(--color-ink-muted)]">
          / {String(total).padStart(2, "0")}
        </span>
      </motion.div>

      {/* Etiqueta vertical institucional */}
      <div className="rotate-180 font-mono text-[10px] tracking-[0.32em] text-[var(--color-ink-muted)] uppercase [writing-mode:vertical-rl]">
        Hespérides · Métodos numéricos
      </div>
    </aside>
  );
}
