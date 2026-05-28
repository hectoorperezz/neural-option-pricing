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
import { Math } from "../components/Math";
import { cases as content } from "../content";

export function CasesSlide() {
  return (
    <>
      <Eyebrow>{content.eyebrow}</Eyebrow>
      <Headline delay={0.08}>
        <RichText fragments={content.headline} />
      </Headline>

      <Body delay={0.22} className="max-w-[64ch]">
        {content.body}
      </Body>

      <Stagger
        className="mt-12 grid flex-1 grid-cols-1 gap-6 sm:grid-cols-2"
        delay={0.4}
      >
        {content.cards.map((card) => (
          <motion.div
            key={card.title}
            variants={fadeUp}
            className={`flex flex-col rounded-3xl border p-8 ${
              card.accent
                ? "border-[var(--color-uni-black)]"
                : "border-black/10 bg-white"
            }`}
            style={
              card.accent
                ? { background: "var(--color-uni-yellow)" }
                : undefined
            }
          >
            <div className="eyebrow mb-4">{card.eyebrow}</div>
            <div className="font-display text-3xl font-medium text-[var(--color-uni-black)]">
              {card.title}
            </div>
            {card.formula && (
              <div
                className={`mt-5 rounded-lg px-4 py-3 text-[15px] ${
                  card.accent
                    ? "bg-[var(--color-uni-black)] text-[var(--color-uni-yellow)]"
                    : "bg-[var(--color-uni-black)] text-white"
                }`}
              >
                <Math>{card.formula}</Math>
              </div>
            )}
            <ul
              className={`mt-6 space-y-2 text-[15px] leading-relaxed ${
                card.accent
                  ? "text-[var(--color-uni-black)]/85"
                  : "text-[var(--color-ink-soft)]"
              }`}
            >
              {card.bullets.map((bullet, i) => (
                <li key={i}>
                  <InlineMathText>{bullet}</InlineMathText>
                </li>
              ))}
            </ul>
          </motion.div>
        ))}
      </Stagger>

      <FooterStamp />
    </>
  );
}
