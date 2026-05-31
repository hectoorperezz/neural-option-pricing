import { useCallback, useEffect, useState } from "react";

export type Direction = 1 | -1;

interface NavigationState {
  index: number;
  /** Paso de revelado dentro de la diapositiva actual (0 = inicial). */
  step: number;
  direction: Direction;
  total: number;
  goNext: () => void;
  goPrev: () => void;
  goTo: (target: number) => void;
}

/**
 * Estado central de la navegación. Además de saltar entre diapositivas,
 * gestiona un revelado por pasos dentro de cada una: avanzar revela el
 * siguiente bloque y, al agotar los pasos, pasa a la diapositiva siguiente.
 *
 * @param stepsPerSlide número de pasos extra (reveals) por diapositiva.
 */
export function useSlideNavigation(stepsPerSlide: number[]): NavigationState {
  const total = stepsPerSlide.length;
  const [index, setIndex] = useState<number>(() => readHashIndex(total));
  const [step, setStep] = useState<number>(0);
  const [direction, setDirection] = useState<Direction>(1);

  const goTo = useCallback(
    (target: number) => {
      const clamped = Math.max(0, Math.min(total - 1, target));
      setDirection(clamped >= index ? 1 : -1);
      setIndex(clamped);
      setStep(0);
      if (typeof window !== "undefined") {
        window.history.replaceState(null, "", `#${clamped + 1}`);
      }
    },
    [index, total],
  );

  const goNext = useCallback(() => {
    const maxStep = stepsPerSlide[index] ?? 0;
    if (step < maxStep) {
      setDirection(1);
      setStep(step + 1);
      return;
    }
    if (index < total - 1) {
      setDirection(1);
      setIndex(index + 1);
      setStep(0);
      if (typeof window !== "undefined") {
        window.history.replaceState(null, "", `#${index + 2}`);
      }
    }
  }, [index, step, stepsPerSlide, total]);

  const goPrev = useCallback(() => {
    if (step > 0) {
      setDirection(-1);
      setStep(step - 1);
      return;
    }
    if (index > 0) {
      setDirection(-1);
      setIndex(index - 1);
      setStep(stepsPerSlide[index - 1] ?? 0); // aterriza con la slide anterior ya revelada
      if (typeof window !== "undefined") {
        window.history.replaceState(null, "", `#${index}`);
      }
    }
  }, [index, step, stepsPerSlide]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const key = event.key;
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
    const handler = () => {
      setIndex(readHashIndex(total));
      setStep(0);
    };
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, [total]);

  return { index, step, direction, total, goNext, goPrev, goTo };
}

function readHashIndex(total: number): number {
  if (typeof window === "undefined") return 0;
  const raw = window.location.hash.replace("#", "");
  const parsed = parseInt(raw, 10);
  if (Number.isNaN(parsed)) return 0;
  return Math.max(0, Math.min(total - 1, parsed - 1));
}
