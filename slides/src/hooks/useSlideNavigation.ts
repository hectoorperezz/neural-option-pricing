import { useCallback, useEffect, useState } from "react";

export type Direction = 1 | -1;

interface NavigationState {
  index: number;
  direction: Direction;
  total: number;
  goNext: () => void;
  goPrev: () => void;
  goTo: (target: number) => void;
}

/**
 * Estado central de la navegación de diapositivas.
 * Permite saltos por teclado, sincronización con el hash de la URL y
 * exposición de `direction` para las transiciones de Framer Motion.
 */
export function useSlideNavigation(total: number): NavigationState {
  const [index, setIndex] = useState<number>(() => readHashIndex(total));
  const [direction, setDirection] = useState<Direction>(1);

  const goTo = useCallback(
    (target: number) => {
      const clamped = Math.max(0, Math.min(total - 1, target));
      if (clamped === index) return;
      setDirection(clamped > index ? 1 : -1);
      setIndex(clamped);
      if (typeof window !== "undefined") {
        window.history.replaceState(null, "", `#${clamped + 1}`);
      }
    },
    [index, total],
  );

  const goNext = useCallback(() => goTo(index + 1), [index, goTo]);
  const goPrev = useCallback(() => goTo(index - 1), [index, goTo]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const key = event.key;
      const meta = event.metaKey || event.ctrlKey;
      if (meta) return;

      switch (key) {
        case "ArrowRight":
        case "ArrowDown":
        case "PageDown":
        case " ":
          event.preventDefault();
          goNext();
          break;
        case "ArrowLeft":
        case "ArrowUp":
        case "PageUp":
          event.preventDefault();
          goPrev();
          break;
        case "Home":
          event.preventDefault();
          goTo(0);
          break;
        case "End":
          event.preventDefault();
          goTo(total - 1);
          break;
        default:
          if (/^[0-9]$/.test(key)) {
            const target = parseInt(key, 10) - 1;
            if (target >= 0 && target < total) {
              event.preventDefault();
              goTo(target);
            }
          }
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [goNext, goPrev, goTo, total]);

  useEffect(() => {
    const handler = () => setIndex(readHashIndex(total));
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, [total]);

  return { index, direction, total, goNext, goPrev, goTo };
}

function readHashIndex(total: number): number {
  if (typeof window === "undefined") return 0;
  const raw = window.location.hash.replace("#", "");
  const parsed = parseInt(raw, 10);
  if (Number.isNaN(parsed)) return 0;
  return Math.max(0, Math.min(total - 1, parsed - 1));
}
