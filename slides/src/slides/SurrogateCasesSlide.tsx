import { motion } from "framer-motion";
import { Body, Eyebrow, FooterStamp, Headline } from "../components/SlideLayout";
import { InlineMathText, RichText } from "../components/RichText";
import { Math } from "../components/Math";
import { surrogate, cases } from "../content";

/**
 * Slide combinada: primero el concepto (una red entrenada offline sustituye al
 * solver) con su pipeline de cuatro pasos, y debajo los dos casos de estudio
 * (Black-Scholes como validación y Heston como caso principal).
 */
export function SurrogateCasesSlide() {
  return (
    <>
      <Eyebrow>{surrogate.eyebrow}</Eyebrow>

      <Headline delay={0.08} size="md">
        <RichText fragments={surrogate.headline} />
      </Headline>

      <Body delay={0.2} className="max-w-[82ch]">
        {surrogate.body}
      </Body>

      {/* Pipeline del surrogate + relación que aprende */}
      <div className="mt-6 grid grid-cols-1 items-center gap-5 lg:grid-cols-[1fr_auto]">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {surrogate.steps.map((step, idx) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                delay: 0.34 + idx * 0.1,
                duration: 0.55,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="rounded-2xl border border-black/10 bg-white p-4"
            >
              <div className="mb-2 flex items-center gap-2">
                <span
                  className="inline-block h-2 w-2"
                  style={{
                    background:
                      idx === 0 || idx === surrogate.steps.length - 1
                        ? "var(--color-uni-yellow)"
                        : "var(--color-uni-black)",
                  }}
                />
                <span className="font-mono text-[11px] tabular-nums text-[var(--color-ink-muted)]">
                  {String(idx + 1).padStart(2, "0")}
                </span>
              </div>
              <div className="font-display text-[15px] font-medium leading-tight text-[var(--color-uni-black)]">
                {step.title}
              </div>
              <div className="mt-1 text-[13px] italic text-[var(--color-ink-muted)]">
                <InlineMathText>{step.subtitle}</InlineMathText>
              </div>
            </motion.div>
          ))}
        </div>

        {surrogate.formula && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8, duration: 0.6 }}
            className="flex flex-col items-center gap-1 rounded-xl border border-black/10 bg-white px-6 py-3 text-center lg:min-w-[18rem]"
          >
            <div className="text-lg text-[var(--color-uni-black)]">
              <Math display>{surrogate.formula.body}</Math>
            </div>
            <div className="max-w-[26rem] text-[12px] italic leading-snug text-[var(--color-ink-muted)]">
              {surrogate.formula.caption}
            </div>
          </motion.div>
        )}
      </div>

      {/* Casos de estudio */}
      <div className="mt-7 flex flex-1 flex-col justify-center border-t border-black/10 pt-5">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9, duration: 0.5 }}
          className="font-display text-[clamp(1rem,1.4vw,1.35rem)] leading-snug text-[var(--color-uni-black)]"
        >
          <RichText fragments={cases.headline} />
        </motion.div>

        <motion.div
          initial="hidden"
          animate="show"
          variants={{
            hidden: {},
            show: { transition: { delayChildren: 1, staggerChildren: 0.12 } },
          }}
          className="mt-5 grid w-full grid-cols-1 gap-5 sm:grid-cols-2"
        >
          {cases.cards.map((card) => (
            <motion.div
              key={card.title}
              variants={{
                hidden: { opacity: 0, y: 14 },
                show: {
                  opacity: 1,
                  y: 0,
                  transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] },
                },
              }}
              className={`flex flex-col rounded-3xl border p-7 ${
                card.accent
                  ? "border-[var(--color-uni-black)]"
                  : "border-black/10 bg-white"
              }`}
              style={
                card.accent ? { background: "var(--color-uni-yellow)" } : undefined
              }
            >
              <div className="flex items-baseline justify-between">
                <div className="font-display text-2xl font-medium text-[var(--color-uni-black)]">
                  {card.title}
                </div>
                <div className="eyebrow">{card.eyebrow}</div>
              </div>
              {card.formula && (
                <div
                  className={`mt-5 overflow-x-auto rounded-lg px-4 py-3 text-[15px] ${
                    card.accent
                      ? "bg-[var(--color-uni-black)] text-[var(--color-uni-yellow)]"
                      : "bg-[var(--color-uni-black)] text-white"
                  }`}
                >
                  <Math>{card.formula}</Math>
                </div>
              )}
              <ul
                className={`mt-5 space-y-2.5 text-[14.5px] leading-relaxed ${
                  card.accent
                    ? "text-[var(--color-uni-black)]/85"
                    : "text-[var(--color-ink-soft)]"
                }`}
              >
                {card.bullets.map((bullet, i) => (
                  <li key={i} className="flex gap-2.5">
                    <span
                      aria-hidden
                      className="mt-[8px] inline-block h-[5px] w-[5px] shrink-0"
                      style={{
                        background: card.accent
                          ? "var(--color-uni-black)"
                          : "var(--color-uni-yellow)",
                      }}
                    />
                    <span>
                      <InlineMathText>{bullet}</InlineMathText>
                    </span>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </motion.div>
      </div>

      <FooterStamp />
    </>
  );
}
