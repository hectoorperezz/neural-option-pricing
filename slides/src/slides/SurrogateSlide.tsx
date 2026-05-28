import { motion } from "framer-motion";
import { Body, Eyebrow, FooterStamp, Headline } from "../components/SlideLayout";
import { InlineMathText, RichText } from "../components/RichText";
import { Math } from "../components/Math";
import { surrogate as content } from "../content";

export function SurrogateSlide() {
  return (
    <>
      <Eyebrow>{content.eyebrow}</Eyebrow>

      <Headline delay={0.08}>
        <RichText fragments={content.headline} />
      </Headline>

      <Body delay={0.22} className="max-w-[60ch]">
        {content.body}
      </Body>

      <div className="mt-10 flex flex-1 flex-col justify-center">
        <div className="grid w-full grid-cols-4 gap-3">
          {content.steps.map((step, idx) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                delay: 0.4 + idx * 0.12,
                duration: 0.6,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="relative"
            >
              <div className="rounded-2xl border border-black/10 bg-white p-5 shadow-[0_2px_18px_rgba(0,0,0,0.04)]">
                <div className="mb-3 flex items-center gap-2">
                  <span
                    className="inline-block h-2 w-2"
                    style={{
                      background:
                        idx === 0 || idx === content.steps.length - 1
                          ? "var(--color-uni-yellow)"
                          : "var(--color-uni-black)",
                    }}
                  />
                  <span className="text-xs font-mono tabular-nums text-[var(--color-ink-muted)]">
                    {String(idx + 1).padStart(2, "0")}
                  </span>
                </div>
                <div className="font-display text-lg font-medium text-[var(--color-uni-black)]">
                  {step.title}
                </div>
                <div className="mt-1 text-sm italic text-[var(--color-ink-muted)]">
                  <InlineMathText>{step.subtitle}</InlineMathText>
                </div>
              </div>
              {idx < content.steps.length - 1 && (
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{
                    delay: 0.55 + idx * 0.12,
                    duration: 0.45,
                    ease: [0.22, 1, 0.36, 1],
                  }}
                  className="pointer-events-none absolute right-[-12px] top-1/2 hidden h-px w-3 origin-left bg-[var(--color-ink-muted)]/50 sm:block"
                />
              )}
            </motion.div>
          ))}
        </div>

        {content.formula && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.95, duration: 0.7 }}
            className="mt-8 flex flex-col items-center gap-2 rounded-xl border border-black/10 bg-white px-6 py-4 text-center"
          >
            <div className="text-xl text-[var(--color-uni-black)]">
              <Math display>{content.formula.body}</Math>
            </div>
            <div className="text-[15px] italic text-[var(--color-ink-muted)] max-w-[68ch]">
              {content.formula.caption}
            </div>
          </motion.div>
        )}
      </div>

      <FooterStamp />
    </>
  );
}
