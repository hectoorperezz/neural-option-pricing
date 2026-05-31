import { motion } from "framer-motion";
import { Body, Eyebrow, FooterStamp, Headline } from "../components/SlideLayout";
import { InlineMathText, RichText } from "../components/RichText";
import { Reveal } from "../components/Reveal";
import { training as content } from "../content";

export function TrainingSlide() {
  return (
    <>
      <Eyebrow>{content.eyebrow}</Eyebrow>
      <Headline delay={0.08} size="md">
        <RichText fragments={content.headline} />
      </Headline>
      <Body delay={0.2} className="max-w-[80ch]">
        {content.body}
      </Body>

      <div className="mt-6 grid flex-1 grid-cols-1 items-stretch gap-8 lg:grid-cols-[0.92fr_1.08fr]">
        {/* Fotos del montaje */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.26, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="grid grid-cols-2 gap-4"
        >
          {content.photos.map((photo) => (
            <figure key={photo.src} className="m-0 flex flex-col">
              <div className="flex-1 overflow-hidden rounded-2xl border border-black/10 bg-white p-2">
                <img
                  src={photo.src}
                  alt={photo.alt}
                  className="h-full w-full rounded-lg object-cover"
                />
              </div>
              <figcaption className="mt-2 text-[11.5px] leading-snug text-[var(--color-ink-muted)]">
                <InlineMathText>{photo.caption}</InlineMathText>
              </figcaption>
            </figure>
          ))}
        </motion.div>

        {/* Bloques de información */}
        <div className="flex flex-col gap-4">
          {content.blocks.map((block, idx) => (
            <Reveal at={idx + 1} key={block.title}>
              <div className="rounded-2xl border border-black/10 bg-white px-6 py-5">
                <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--color-ink-muted)]">
                  <span
                    aria-hidden
                    className="inline-block h-[6px] w-[6px]"
                    style={{ background: "var(--color-uni-yellow)" }}
                  />
                  {block.title}
                </div>
                <ul className="mt-3 space-y-2 text-[15px] leading-relaxed text-[var(--color-ink-soft)]">
                  {block.bullets.map((bullet, i) => (
                    <li key={i} className="flex gap-2.5">
                      <span
                        aria-hidden
                        className="mt-[8px] inline-block h-[5px] w-[5px] shrink-0"
                        style={{ background: "var(--color-uni-black)" }}
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
      </div>

      <FooterStamp />
    </>
  );
}
