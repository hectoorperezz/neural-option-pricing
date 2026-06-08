import { motion } from "framer-motion";
import {
  Body,
  Eyebrow,
  FooterStamp,
  Headline,
  Stagger,
  fadeUp,
} from "../components/SlideLayout";
import { InlineMathText, RichText } from "../components/RichText";
import { literature as content } from "../content";

export function LiteratureSlide() {
  return (
    <>
      <Eyebrow>{content.eyebrow}</Eyebrow>
      <Headline delay={0.08}>
        <RichText fragments={content.headline} />
      </Headline>
      <Body delay={0.18} className="max-w-[68ch]">
        {content.body}
      </Body>

      <Stagger className="mt-7 flex flex-1 flex-col gap-3" delay={0.32}>
        {content.papers.map((paper, idx) => (
          <motion.div
            key={paper.authors}
            variants={fadeUp}
            className={`grid grid-cols-[3.5rem_minmax(0,1fr)_minmax(0,2.2fr)_minmax(0,12rem)] items-center gap-5 rounded-2xl border px-5 py-3.5 ${
              paper.central
                ? "border-[var(--color-uni-black)] bg-white shadow-[0_4px_16px_rgba(0,0,0,0.05)]"
                : "border-black/10 bg-white"
            }`}
          >
            {/* Índice */}
            <div className="flex flex-col items-start">
              <span className="font-mono text-[12px] tracking-[0.22em] text-[var(--color-ink-muted)]">
                {String(idx + 1).padStart(2, "0")}
              </span>
              {paper.central && (
                <span
                  className="mt-1 inline-block h-[6px] w-[18px]"
                  style={{ background: "var(--color-uni-yellow)" }}
                />
              )}
            </div>

            {/* Autores + año */}
            <div>
              <div className="font-display text-[18px] font-medium leading-tight text-[var(--color-uni-black)]">
                {paper.authors}
              </div>
              <div className="mt-0.5 font-mono text-[12px] text-[var(--color-ink-muted)]">
                {paper.year}
              </div>
            </div>

            {/* Título + contribución */}
            <div>
              <div className="text-[16px] font-medium leading-snug text-[var(--color-uni-black)]">
                {paper.title}
              </div>
              <div className="mt-1 text-[14px] leading-snug text-[var(--color-ink-soft)]">
                <InlineMathText>{paper.contribution}</InlineMathText>
              </div>
            </div>

            {/* Rol */}
            <div
              className={`justify-self-end rounded-full px-3 py-1 text-center text-[11px] font-semibold uppercase tracking-[0.18em] ${
                paper.central
                  ? "bg-[var(--color-uni-yellow)] text-[var(--color-uni-black)]"
                  : "bg-black/5 text-[var(--color-ink-soft)]"
              }`}
            >
              {paper.role}
            </div>
          </motion.div>
        ))}
      </Stagger>

      <FooterStamp />
    </>
  );
}
