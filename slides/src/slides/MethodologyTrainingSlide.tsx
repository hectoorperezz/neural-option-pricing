import { motion } from "framer-motion";
import { Eyebrow, FooterStamp, Headline } from "../components/SlideLayout";
import { InlineMathText, RichText } from "../components/RichText";
import { Math } from "../components/Math";
import { Reveal } from "../components/Reveal";
import { methodology, training } from "../content";

/**
 * Slide combinada: el pipeline metodológico (dominio y muestreo, targets y
 * métricas, evaluación por bins) y, debajo, el montaje físico con el que se
 * entrenó y escaló todo (hardware + fotos de la estación de trabajo).
 */
export function MethodologyTrainingSlide() {
  return (
    <>
      <Eyebrow>{methodology.eyebrow}</Eyebrow>
      <Headline delay={0.08} size="md">
        <RichText fragments={methodology.headline} />
      </Headline>

      {/* Pipeline metodológico */}
      <div className="mt-6 grid grid-cols-1 items-stretch gap-5 lg:grid-cols-3">
        {methodology.blocks.map((block, idx) => (
          <Reveal at={idx + 1} key={block.title} className="flex">
            <div
              className={`flex w-full flex-col rounded-2xl border p-5 ${
                block.accent
                  ? "border-[var(--color-uni-black)]"
                  : "border-black/10 bg-white"
              }`}
              style={
                block.accent ? { background: "var(--color-uni-yellow)" } : undefined
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
              <div className="font-display mt-1.5 text-xl font-medium text-[var(--color-uni-black)]">
                {block.title}
              </div>
              {block.formula && (
                <div
                  className={`mt-3 overflow-x-auto rounded-xl px-3 py-2.5 text-[14px] ${
                    block.accent
                      ? "bg-[var(--color-uni-black)] text-[var(--color-uni-yellow)]"
                      : "bg-[var(--color-uni-black)] text-white"
                  }`}
                >
                  <Math display>{block.formula}</Math>
                </div>
              )}
              <ul
                className={`mt-3 space-y-2 text-[13px] leading-relaxed ${
                  block.accent
                    ? "text-[var(--color-uni-black)]/85"
                    : "text-[var(--color-ink-soft)]"
                }`}
              >
                {block.bullets.map((bullet, i) => (
                  <li key={i} className="flex gap-2.5">
                    <span
                      aria-hidden
                      className="mt-[7px] inline-block h-[5px] w-[5px] shrink-0"
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

      {/* Hardware y montaje físico */}
      <div className="mt-6 flex flex-1 items-stretch gap-6 border-t border-black/10 pt-5">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.55, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="grid w-[46%] shrink-0 grid-cols-2 gap-4"
        >
          {training.photos.map((photo) => (
            <figure key={photo.src} className="m-0 flex flex-col">
              <div className="flex-1 overflow-hidden rounded-2xl border border-black/10 bg-white p-2">
                <img
                  src={photo.src}
                  alt={photo.alt}
                  className="h-full w-full rounded-lg object-cover"
                />
              </div>
              <figcaption className="mt-2 text-[11px] leading-snug text-[var(--color-ink-muted)]">
                <InlineMathText>{photo.caption}</InlineMathText>
              </figcaption>
            </figure>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="flex flex-1 flex-col justify-center gap-3"
        >
          <div className="eyebrow">Hardware y escalado</div>
          <ul className="space-y-2 text-[14px] leading-relaxed text-[var(--color-ink-soft)]">
            {[
              "Estación única: CPU Intel i9-14900K (32 workers), 64 GB de RAM y GPU NVIDIA RTX 4060.",
              "Régimen común: MLP $4\\times 128$ con Swish, Adam $\\mathrm{lr}=10^{-3}$, 100 épocas; los once pre-registrados en $\\approx$63 min.",
              "Escalado $\\times 50$ (datasets de hasta 25M): el error cayó de $\\approx 10^{-2}$ a $\\approx 10^{-3}$ con la arquitectura fija.",
            ].map((item, i) => (
              <li key={i} className="flex gap-2.5">
                <span
                  aria-hidden
                  className="mt-[8px] inline-block h-[5px] w-[5px] shrink-0"
                  style={{ background: "var(--color-uni-yellow)" }}
                />
                <span>
                  <InlineMathText>{item}</InlineMathText>
                </span>
              </li>
            ))}
          </ul>
        </motion.div>
      </div>

      <FooterStamp />
    </>
  );
}
