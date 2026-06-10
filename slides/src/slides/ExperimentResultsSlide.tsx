import { motion } from "framer-motion";
import { Eyebrow, FooterStamp, Headline } from "../components/SlideLayout";
import { InlineMathText } from "../components/RichText";
import { Reveal } from "../components/Reveal";

export interface ExperimentResultsSlideProps {
  id: string;
  title: string;
  figure: { src: string; alt: string; caption: string };
  findings: string[];
  conclusion: string;
}

export function ExperimentResultsSlide({
  id,
  title,
  figure,
  findings,
  conclusion,
}: ExperimentResultsSlideProps) {
  return (
    <>
      <Eyebrow>{`Experimento ${id} · Resultados`}</Eyebrow>
      <Headline delay={0.08} size="md">
        {title}
      </Headline>

      <div className="mt-5 grid flex-1 grid-cols-1 items-center gap-10 lg:grid-cols-[1.15fr_1fr]">
        {/* Figura */}
        <motion.figure
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.18, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="m-0 flex flex-col"
        >
          <div className="rounded-2xl border border-black/10 bg-white p-4 shadow-sm">
            <img
              src={figure.src}
              alt={figure.alt}
              className="mx-auto h-auto w-full max-h-[64vh] object-contain"
            />
          </div>
          <figcaption className="mt-3 max-w-[60ch] text-[13px] leading-snug text-[var(--color-ink-muted)]">
            <InlineMathText>{figure.caption}</InlineMathText>
          </figcaption>
        </motion.figure>

        {/* Hallazgos + conclusión */}
        <div className="flex flex-col">
          <div className="text-[12px] font-semibold uppercase tracking-[0.22em] text-[var(--color-ink-muted)]">
            Hallazgos
          </div>
          <ul className="mt-4 space-y-[1.15rem] text-[clamp(1rem,1.3vw,1.25rem)] leading-[1.45] text-[var(--color-uni-black)]">
            {findings.map((item, i) => (
              <Reveal at={i + 1} key={i}>
                <li className="flex gap-3">
                  <span
                    aria-hidden
                    className="mt-[9px] inline-block h-[7px] w-[7px] shrink-0"
                    style={{ background: "var(--color-uni-yellow)" }}
                  />
                  <span>
                    <InlineMathText>{item}</InlineMathText>
                  </span>
                </li>
              </Reveal>
            ))}
          </ul>

          <Reveal
            at={findings.length + 1}
            className="mt-7 rounded-xl border-l-4 border-[var(--color-uni-black)] bg-[var(--color-uni-yellow)] px-6 py-5"
          >
            <div className="text-[12px] font-semibold uppercase tracking-[0.22em] text-[var(--color-uni-black)]/70">
              Conclusión
            </div>
            <div className="mt-2 text-[clamp(1rem,1.3vw,1.25rem)] leading-[1.45] text-[var(--color-uni-black)]">
              <InlineMathText>{conclusion}</InlineMathText>
            </div>
          </Reveal>
        </div>
      </div>

      <FooterStamp />
    </>
  );
}
