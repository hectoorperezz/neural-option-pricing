import { motion } from "framer-motion";
import { InlineMathText, RichText } from "../components/RichText";
import { closing as content } from "../content";

export function ClosingSlide() {
  return (
    <div className="relative flex h-full w-full overflow-hidden">
      {/* Bandas amarillas */}
      <div
        aria-hidden
        className="absolute inset-x-0 top-0 z-0 h-[6.5%]"
        style={{ background: "var(--color-uni-yellow)" }}
      />
      <div
        aria-hidden
        className="absolute inset-x-0 bottom-0 z-0 h-[4.5%]"
        style={{ background: "var(--color-uni-yellow)" }}
      />

      {/* Monograma .h gigante a sangre por la izquierda */}
      <motion.img
        aria-hidden
        src="/h-mono.png"
        alt=""
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
        className="absolute -left-[8%] top-1/2 z-0 h-[80%] -translate-y-1/2"
      />

      {/* Halftone amarillo en la esquina inferior derecha */}
      <motion.svg
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 1 }}
        viewBox="0 0 100 100"
        preserveAspectRatio="xMaxYMax meet"
        className="absolute bottom-0 right-0 z-0 h-[40%] w-[30%]"
      >
        {Array.from({ length: 9 }).map((_, i) =>
          Array.from({ length: 7 }).map((_, j) => {
            const cx = 100 - i * 11;
            const cy = 100 - j * 11;
            const dist = Math.hypot(i, j);
            const r = Math.max(0, 4.3 - dist * 0.32);
            if (r <= 0.1) return null;
            return (
              <circle
                key={`${i}-${j}`}
                cx={cx}
                cy={cy}
                r={r}
                fill="var(--color-uni-yellow)"
              />
            );
          }),
        )}
      </motion.svg>

      {/* Contenido central */}
      <div className="relative z-10 flex h-full w-full flex-col items-center justify-center px-12 text-center">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="eyebrow"
        >
          {content.eyebrow}
        </motion.div>

        <motion.h2
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
          className="font-display mt-6 text-[clamp(2.4rem,5vw,4.6rem)] leading-[1.05] text-[var(--color-uni-black)]"
        >
          <RichText fragments={content.headline} />
        </motion.h2>

        <motion.div
          initial={{ opacity: 0, scaleX: 0 }}
          animate={{ opacity: 1, scaleX: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="mt-8 h-[3px] w-20"
          style={{ background: "var(--color-uni-black)" }}
        />

        {content.takeaways && content.takeaways.length > 0 && (
          <motion.ul
            initial="hidden"
            animate="show"
            variants={{
              hidden: {},
              show: {
                transition: { delayChildren: 0.7, staggerChildren: 0.12 },
              },
            }}
            className="mt-8 flex max-w-[62ch] flex-col gap-2.5 text-left text-[16px] text-[var(--color-ink-soft)]"
          >
            {content.takeaways.map((item, idx) => (
              <motion.li
                key={idx}
                variants={{
                  hidden: { opacity: 0, y: 8 },
                  show: {
                    opacity: 1,
                    y: 0,
                    transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] },
                  },
                }}
                className="flex items-start gap-3"
              >
                <span
                  className="mt-2 inline-block h-[6px] w-[6px] shrink-0"
                  style={{ background: "var(--color-uni-yellow)" }}
                />
                <span className="leading-snug">
                  <InlineMathText>{item}</InlineMathText>
                </span>
              </motion.li>
            ))}
          </motion.ul>
        )}

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.0, duration: 0.7 }}
          className="mt-8 flex flex-col items-center gap-2 text-xs uppercase tracking-[0.22em] text-[var(--color-ink-muted)]"
        >
          <span>{content.footer.institution}</span>
          <span className="font-display normal-case tracking-normal italic text-[var(--color-ink-soft)]">
            {content.footer.subject}
          </span>
        </motion.div>
      </div>
    </div>
  );
}
