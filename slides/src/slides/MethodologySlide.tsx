import {
  Body,
  Eyebrow,
  FooterStamp,
  Headline,
} from "../components/SlideLayout";
import { InlineMathText, RichText } from "../components/RichText";
import { Math } from "../components/Math";
import { Reveal } from "../components/Reveal";
import { methodology as content } from "../content";

export function MethodologySlide() {
  return (
    <>
      <Eyebrow>{content.eyebrow}</Eyebrow>
      <Headline delay={0.08} size="md">
        <RichText fragments={content.headline} />
      </Headline>

      <Body delay={0.2} className="max-w-[78ch]">
        {content.body}
      </Body>

      <div className="mt-8 grid flex-1 grid-cols-1 items-stretch gap-6 lg:grid-cols-3">
        {content.blocks.map((block, idx) => (
          <Reveal at={idx + 1} key={block.title} className="flex">
            <div
              className={`flex w-full flex-col rounded-2xl border p-7 ${
                block.accent
                  ? "border-[var(--color-uni-black)]"
                  : "border-black/10 bg-white"
              }`}
              style={
                block.accent
                  ? { background: "var(--color-uni-yellow)" }
                  : undefined
              }
            >
              <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--color-ink-muted)]">
                <span
                  aria-hidden
                  className="inline-block h-[6px] w-[6px]"
                  style={{ background: "var(--color-uni-black)" }}
                />
                Paso {idx + 1}
              </div>
              <div className="font-display mt-2 text-2xl font-medium text-[var(--color-uni-black)]">
                {block.title}
              </div>
              {block.formula && (
                <div
                  className={`mt-5 overflow-x-auto rounded-xl px-4 py-3 text-[16px] ${
                    block.accent
                      ? "bg-[var(--color-uni-black)] text-[var(--color-uni-yellow)]"
                      : "bg-[var(--color-uni-black)] text-white"
                  }`}
                >
                  <Math display>{block.formula}</Math>
                </div>
              )}
              <ul
                className={`mt-5 space-y-3 text-[15px] leading-relaxed ${
                  block.accent
                    ? "text-[var(--color-uni-black)]/85"
                    : "text-[var(--color-ink-soft)]"
                }`}
              >
                {block.bullets.map((bullet, i) => (
                  <li key={i} className="flex gap-2.5">
                    <span
                      aria-hidden
                      className="mt-[8px] inline-block h-[5px] w-[5px] shrink-0"
                      style={{
                        background: block.accent
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
            </div>
          </Reveal>
        ))}
      </div>

      <FooterStamp />
    </>
  );
}
