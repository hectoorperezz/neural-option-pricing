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

      <div className="mt-12 grid flex-1 grid-cols-1 gap-3 sm:grid-cols-5">
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
            className="group relative flex flex-col rounded-2xl border border-black/10 p-5 transition hover:border-[var(--color-uni-black)]"
          >
            <div className="flex items-baseline gap-2">
              <span
                className="inline-block h-[6px] w-[6px]"
                style={{ background: "var(--color-uni-yellow)" }}
              />
              <span className="font-mono text-xs tracking-widest text-[var(--color-ink-muted)]">
                {exp.id}
              </span>
            </div>
            <div className="font-display mt-3 text-lg font-medium text-[var(--color-uni-black)]">
              {exp.title}
            </div>
            <div className="mt-3 text-[15px] leading-relaxed text-[var(--color-ink-soft)]">
              {exp.summary}
            </div>
            {exp.metric && (
              <div className="mt-auto pt-4">
                <div className="text-[11px] uppercase tracking-[0.22em] text-[var(--color-ink-muted)]">
                  Métrica
                </div>
                <div className="mt-1 text-[14px] text-[var(--color-uni-black)]">
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
