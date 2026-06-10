import { motion } from "framer-motion";
import { Body, Eyebrow, FooterStamp, Headline } from "../components/SlideLayout";
import { InlineMathText } from "../components/RichText";
import { Reveal } from "../components/Reveal";

export interface ExperimentSlideProps {
  id: string;
  title: string;
  question: string;
  /** Métrica primaria. Texto con LaTeX inline (`$...$`). */
  metric: string;
  surrogates: string[];
  /** Hipótesis previa al experimento. */
  hypothesis?: string;
}

export function ExperimentSlide({
  id,
  title,
  question,
  metric,
  surrogates,
  hypothesis,
}: ExperimentSlideProps) {
  return (
    <>
      <Eyebrow>{`Experimento ${id} · Planteamiento`}</Eyebrow>

      <div className="relative flex flex-1 flex-col justify-center">
        {/* Identificador gigante como marca de agua, llena el flanco derecho */}
        <motion.div
          aria-hidden
          initial={{ opacity: 0, x: 40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.25, duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
          className="pointer-events-none absolute right-0 top-1/2 hidden -translate-y-1/2 select-none font-display text-[24rem] font-semibold leading-none tracking-tighter text-black/[0.045] lg:block"
        >
          {id}
        </motion.div>

        <Headline delay={0.08} size="lg">
          {title}
        </Headline>
        <Body
          delay={0.18}
          className="max-w-[64ch] text-[clamp(1.1rem,1.5vw,1.55rem)] leading-[1.5]"
        >
          <InlineMathText>{question}</InlineMathText>
        </Body>

        {/* Ficha del experimento: métrica primaria + surrogates implicados */}
        <Reveal at={1} className="mt-10 flex flex-wrap items-stretch gap-4">
          <div className="flex min-w-[22rem] flex-col justify-center rounded-2xl border border-black/10 bg-white px-7 py-4 shadow-[0_2px_14px_rgba(0,0,0,0.04)]">
            <div className="flex items-center gap-2 text-[11.5px] font-semibold uppercase tracking-[0.22em] text-[var(--color-ink-muted)]">
              <span
                aria-hidden
                className="inline-block h-[6px] w-[6px]"
                style={{ background: "var(--color-uni-yellow)" }}
              />
              Métrica primaria
            </div>
            <div className="mt-2 text-[17px] text-[var(--color-uni-black)]">
              <InlineMathText>{metric}</InlineMathText>
            </div>
          </div>

          <div className="flex flex-col justify-center rounded-2xl bg-[var(--color-uni-black)] px-7 py-4">
            <div className="text-[11.5px] font-semibold uppercase tracking-[0.22em] text-white/60">
              Surrogates
            </div>
            <div className="mt-2.5 flex flex-wrap items-center gap-2">
              {surrogates.map((s) => (
                <span
                  key={s}
                  className="rounded-md border border-white/25 px-2.5 py-1 font-mono text-[14px] leading-none text-white"
                >
                  {s}
                </span>
              ))}
            </div>
          </div>
        </Reveal>

        {hypothesis && (
          <Reveal
            at={2}
            className="mt-6 max-w-[78ch] rounded-2xl border-l-4 border-[var(--color-uni-yellow)] bg-white px-8 py-6 shadow-[0_2px_14px_rgba(0,0,0,0.04)]"
          >
            <div className="text-[12px] font-semibold uppercase tracking-[0.22em] text-[var(--color-ink-muted)]">
              Hipótesis previa
            </div>
            <div className="mt-3 text-[clamp(1.05rem,1.4vw,1.4rem)] leading-[1.55] text-[var(--color-uni-black)]">
              <InlineMathText>{hypothesis}</InlineMathText>
            </div>
          </Reveal>
        )}
      </div>

      <FooterStamp />
    </>
  );
}
