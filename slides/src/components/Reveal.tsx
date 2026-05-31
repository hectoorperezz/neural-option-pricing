import { createContext, useContext } from "react";
import { motion } from "framer-motion";
import type { ReactNode } from "react";

/** Paso actual dentro de la diapositiva (0 = estado inicial). */
export const StepContext = createContext(0);
export const useStep = () => useContext(StepContext);

interface RevealProps {
  /** Paso a partir del cual el componente es visible. 0 = visible de inicio. */
  at?: number;
  children: ReactNode;
  className?: string;
  /** Desplazamiento vertical de entrada (px). */
  y?: number;
}

/**
 * Envuelve un bloque para que aparezca cuando el paso actual de la
 * diapositiva alcanza `at`. El espacio se reserva siempre (para que el
 * layout no salte) y el bloque se anima al revelarse.
 */
export function Reveal({ at = 0, children, className = "", y = 16 }: RevealProps) {
  const step = useStep();
  const shown = step >= at;
  return (
    <motion.div
      className={className}
      initial={false}
      animate={{
        opacity: shown ? 1 : 0,
        y: shown ? 0 : y,
        filter: shown ? "blur(0px)" : "blur(4px)",
      }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      style={{ pointerEvents: shown ? "auto" : "none" }}
      aria-hidden={!shown}
    >
      {children}
    </motion.div>
  );
}
