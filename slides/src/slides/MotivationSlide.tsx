import { motion } from "framer-motion";
import {
  Body,
  Eyebrow,
  FooterStamp,
  Headline,
  Stagger,
  fadeUp,
} from "../components/SlideLayout";
import { RichText } from "../components/RichText";
import { motivation as content } from "../content";

export function MotivationSlide() {
  return (
    <>
      <Eyebrow>{content.eyebrow}</Eyebrow>

      <Headline delay={0.08}>
        <RichText fragments={content.headline} />
      </Headline>

      <Body delay={0.22}>{content.body}</Body>

      <Stagger
        className="mt-12 grid grid-cols-1 gap-4 sm:grid-cols-3"
        delay={0.35}
      >
        {content.items.map((item) => (
          <motion.div
            key={item.title}
            variants={fadeUp}
            className={`relative overflow-hidden rounded-2xl border p-6 ${
              item.accent
                ? "border-[var(--color-uni-black)]"
                : "border-black/10"
            }`}
            style={{
              background: item.accent ? "var(--color-uni-yellow)" : "white",
            }}
          >
            <div className="eyebrow mb-3">{item.note}</div>
            <div className="font-display text-2xl font-medium text-[var(--color-uni-black)]">
              {item.title}
            </div>
            <div className="mt-6 font-mono text-sm tabular-nums text-[var(--color-ink-soft)]">
              evaluación {item.cost}
            </div>
          </motion.div>
        ))}
      </Stagger>

      {content.callout && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.7 }}
          className="mt-8 flex items-start gap-3 rounded-xl border-l-[3px] border-[var(--color-uni-yellow)] bg-white px-5 py-3"
        >
          <div className="text-[13px] font-semibold uppercase tracking-[0.22em] text-[var(--color-ink-muted)] shrink-0 max-w-[16rem]">
            {content.callout.kicker}
          </div>
          <div className="text-[16px] leading-snug text-[var(--color-uni-black)]">
            {content.callout.body}
          </div>
        </motion.div>
      )}

      <FooterStamp />
    </>
  );
}
