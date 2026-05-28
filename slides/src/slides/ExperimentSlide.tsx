import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { Body, Eyebrow, FooterStamp, Headline } from "../components/SlideLayout";
import { InlineMathText } from "../components/RichText";

export interface ExperimentSlideProps {
  id: string;
  title: string;
  question: string;
  /** Métrica primaria. Se interpreta como texto con LaTeX inline
   *  (`$...$`), igual que `cards.bullets`. */
  metric: string;
  surrogates: string[];
  /** Hipótesis previa al experimento. */
  hypothesis?: string;
  /** Criterio operativo (solo aplica a E3 y E5). */
  threshold?: string;
  /** Contenido específico (resultados, figuras, tablas). Si está vacío,
   *  se muestra un placeholder discreto. */
  results?: ReactNode;
}

export function ExperimentSlide({
  id,
  title,
  question,
  metric,
  surrogates,
  hypothesis,
  threshold,
  results,
}: ExperimentSlideProps) {
  return (
    <>
      <Eyebrow>{`Experimento ${id}`}</Eyebrow>

      <Headline delay={0.08} size="lg">
        {title}
      </Headline>
      <Body delay={0.18}>{question}</Body>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <div className="inline-flex items-center gap-3 rounded-full border border-black/10 px-4 py-1.5 text-xs uppercase tracking-widest text-[var(--color-ink-soft)]">
          <span
            className="inline-block h-2 w-2"
            style={{ background: "var(--color-uni-yellow)" }}
          />
          Métrica primaria{" "}
          <span className="normal-case tracking-normal">
            <InlineMathText>{metric}</InlineMathText>
          </span>
        </div>
        <div className="inline-flex items-center gap-2 rounded-full bg-[var(--color-uni-black)] px-4 py-1.5 text-xs uppercase tracking-widest text-white">
          Surrogates{" "}
          <span className="font-mono normal-case tracking-normal">
            {surrogates.join(" · ")}
          </span>
        </div>
      </div>

      {(hypothesis || threshold) && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.28, duration: 0.6 }}
          className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2"
        >
          {hypothesis && (
            <div className="rounded-xl border-l-[3px] border-[var(--color-uni-yellow)] bg-white px-5 py-4">
              <div className="text-[12px] font-semibold uppercase tracking-[0.22em] text-[var(--color-ink-muted)]">
                Hipótesis
              </div>
              <div className="mt-2 text-[15px] leading-snug text-[var(--color-uni-black)]">
                <InlineMathText>{hypothesis}</InlineMathText>
              </div>
            </div>
          )}
          {threshold && (
            <div className="rounded-xl border-l-[3px] border-[var(--color-uni-black)] bg-white px-5 py-4">
              <div className="text-[12px] font-semibold uppercase tracking-[0.22em] text-[var(--color-ink-muted)]">
                Criterio operativo
              </div>
              <div className="mt-2 text-[15px] leading-snug text-[var(--color-uni-black)]">
                <InlineMathText>{threshold}</InlineMathText>
              </div>
            </div>
          )}
        </motion.div>
      )}

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        className="mt-8 flex flex-1 flex-col"
      >
        {results ?? <ResultsPlaceholder />}
      </motion.div>

      <FooterStamp />
    </>
  );
}

function ResultsPlaceholder() {
  const slots = [
    { label: "Tabla por bin", icon: <TableIcon /> },
    { label: "Heatmap / curva", icon: <ChartIcon /> },
    { label: "Conclusión", icon: <NoteIcon /> },
  ];

  return (
    <div className="relative grid h-full w-full grid-cols-1 gap-4 sm:grid-cols-3">
      {slots.map((slot, idx) => (
        <div
          key={slot.label}
          className="relative overflow-hidden rounded-2xl border border-black/10 bg-white"
        >
          <div
            className="absolute inset-x-0 top-0 h-[3px]"
            style={{ background: "var(--color-uni-yellow)" }}
          />
          <DotPattern className="absolute inset-0 opacity-[0.20]" />

          <div className="relative flex h-full flex-col items-start justify-between p-6">
            <div className="text-[var(--color-uni-black)]/70">{slot.icon}</div>
            <div>
              <div className="font-mono text-[13px] tracking-[0.25em] text-[var(--color-ink-muted)] uppercase">
                {String(idx + 1).padStart(2, "0")} · pendiente
              </div>
              <div className="font-display mt-2 text-xl font-medium italic text-[var(--color-uni-black)]/85">
                {slot.label}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function TableIcon() {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
      <rect x="3" y="6" width="28" height="22" rx="2" stroke="currentColor" strokeWidth="1.4" />
      <line x1="3" y1="13" x2="31" y2="13" stroke="currentColor" strokeWidth="1.4" />
      <line x1="12" y1="6" x2="12" y2="28" stroke="currentColor" strokeWidth="1.4" />
      <line x1="22" y1="6" x2="22" y2="28" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

function ChartIcon() {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
      <polyline
        points="4,26 11,18 17,22 23,11 30,15"
        stroke="currentColor"
        strokeWidth="1.6"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <line x1="3" y1="30" x2="31" y2="30" stroke="currentColor" strokeWidth="1.2" />
      <line x1="3" y1="30" x2="3" y2="6" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

function NoteIcon() {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
      <rect x="6" y="5" width="22" height="24" rx="2" stroke="currentColor" strokeWidth="1.4" />
      <line x1="10" y1="11" x2="24" y2="11" stroke="currentColor" strokeWidth="1.4" />
      <line x1="10" y1="16" x2="24" y2="16" stroke="currentColor" strokeWidth="1.4" />
      <line x1="10" y1="21" x2="20" y2="21" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

function DotPattern({ className = "" }: { className?: string }) {
  return (
    <svg
      aria-hidden
      className={className}
      preserveAspectRatio="xMidYMid slice"
      viewBox="0 0 200 200"
    >
      <defs>
        <pattern
          id="dot-pattern"
          x="0"
          y="0"
          width="14"
          height="14"
          patternUnits="userSpaceOnUse"
        >
          <circle cx="2" cy="2" r="1.4" fill="var(--color-uni-yellow)" />
        </pattern>
      </defs>
      <rect width="200" height="200" fill="url(#dot-pattern)" />
    </svg>
  );
}
