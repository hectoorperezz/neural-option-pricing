import { motion } from "framer-motion";
import { Eyebrow, FooterStamp, Headline } from "../components/SlideLayout";
import { InlineMathText } from "../components/RichText";
import { experimentsOverview as content } from "../content";

export function ExperimentsOverviewSlide() {
  return (
    <>
      <Eyebrow>{content.eyebrow}</Eyebrow>
      <Headline delay={0.08} size="lg">
        {content.headline}
      </Headline>

      <div className="mt-10 grid flex-1 grid-cols-1 gap-4 sm:grid-cols-3">
        {content.experiments.map((exp, idx) => (
          <motion.div
            key={exp.id}
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              delay: 0.3 + idx * 0.1,
              duration: 0.6,
              ease: [0.22, 1, 0.36, 1],
            }}
            className="group relative flex flex-col rounded-2xl border border-black/10 bg-white p-6 shadow-[0_2px_18px_rgba(0,0,0,0.05)] transition duration-300 hover:-translate-y-1 hover:border-[var(--color-uni-black)] hover:shadow-[0_12px_34px_rgba(0,0,0,0.10)]"
          >
            {/* Cabecera: código del experimento + dimensión */}
            <div className="flex items-center gap-3">
              <span
                className="inline-flex h-8 min-w-[2.6rem] items-center justify-center rounded-lg px-2 font-mono text-[15px] font-bold tracking-wider text-[var(--color-uni-black)]"
                style={{ background: "var(--color-uni-yellow)" }}
              >
                {exp.id}
              </span>
              <span className="text-[12px] font-semibold uppercase tracking-[0.16em] text-[var(--color-ink-muted)]">
                {exp.title}
              </span>
            </div>

            {/* Pregunta de investigación (protagonista) */}
            <div className="flex flex-1 items-center py-6">
              <p className="text-[clamp(1.1rem,1.3vw,1.45rem)] font-medium leading-[1.32] text-[var(--color-uni-black)]">
                {exp.summary}
              </p>
            </div>

            {exp.metric && (
              <div className="border-t border-black/10 pt-4">
                <div className="text-[10.5px] font-semibold uppercase tracking-[0.22em] text-[var(--color-ink-muted)]">
                  Métrica
                </div>
                <div className="mt-1.5 text-[15px] text-[var(--color-uni-black)]">
                  <InlineMathText>{exp.metric}</InlineMathText>
                </div>
              </div>
            )}
          </motion.div>
        ))}
      </div>

      <FooterStamp />
    </>
  );
}
