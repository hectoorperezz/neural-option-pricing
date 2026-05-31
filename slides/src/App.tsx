import { useEffect, useMemo, useState } from "react";
import { SideRail } from "./components/SideRail";
import { SlideBackdrop } from "./components/SlideBackdrop";
import { SlideChrome } from "./components/SlideChrome";
import { SlideStage } from "./components/SlideStage";
import { StepContext } from "./components/Reveal";
import { useSlideNavigation } from "./hooks/useSlideNavigation";
import { slides } from "./slides";

export function App() {
  const stepsPerSlide = useMemo(() => slides.map((s) => s.steps ?? 0), []);
  const { index, step, direction, total, goNext, goPrev } =
    useSlideNavigation(stepsPerSlide);
  const current = slides[index];
  const bleed = current.bleed ?? false;
  const maxStep = current.steps ?? 0;

  // Vista de presentador: panel de notas conmutable con la tecla N.
  const [showNotes, setShowNotes] = useState(false);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "n" || event.key === "N") {
        event.preventDefault();
        setShowNotes((v) => !v);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div
      className="relative h-screen w-screen"
      style={{ background: "var(--color-page)" }}
    >
      {!bleed && <SlideBackdrop />}
      {!bleed && <SideRail index={index} total={total} />}

      <SlideStage index={index} direction={direction} bleed={bleed}>
        <StepContext.Provider value={step}>
          {current.render()}
        </StepContext.Provider>
      </SlideStage>

      <SlideChrome
        index={index}
        total={total}
        onPrev={goPrev}
        onNext={goNext}
        showCounter={bleed}
        step={step}
        maxStep={maxStep}
      />

      {showNotes && (
        <div className="pointer-events-none fixed inset-x-0 bottom-0 z-50 px-6 pb-20">
          <div className="mx-auto max-w-[80ch] rounded-2xl border border-white/10 bg-[var(--color-uni-black)]/95 px-6 py-4 text-white shadow-2xl">
            <div className="mb-1 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--color-uni-yellow)]">
              <span
                aria-hidden
                className="inline-block h-[6px] w-[6px]"
                style={{ background: "var(--color-uni-yellow)" }}
              />
              Notas del orador · {current.label}
            </div>
            <p className="text-[15px] leading-snug text-white/90">
              {current.notes ?? "— (sin notas para esta diapositiva)"}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
