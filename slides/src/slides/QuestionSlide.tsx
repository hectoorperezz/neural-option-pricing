import { motion } from "framer-motion";
import { Eyebrow, FooterStamp } from "../components/SlideLayout";
import { RichText } from "../components/RichText";
import { Math } from "../components/Math";
import { question as content } from "../content";

export function QuestionSlide() {
  return (
    <>
      <Eyebrow>{content.eyebrow}</Eyebrow>

      <div className="mt-10 flex flex-1 flex-col justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="font-display text-[clamp(2rem,4vw,3.6rem)] leading-[1.15] text-[var(--color-uni-black)]"
        >
          {content.lines.map((line, i) => (
            <motion.span
              key={line}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25 + i * 0.2, duration: 0.7 }}
              className="block"
            >
              {line}
            </motion.span>
          ))}
          <motion.span
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 + content.lines.length * 0.2, duration: 0.7 }}
            className="block italic"
          >
            <RichText fragments={content.italicLine} />
          </motion.span>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.0, duration: 0.7 }}
          className="mt-10 max-w-[60ch] text-[clamp(0.95rem,1.15vw,1.1rem)] leading-[1.6] text-[var(--color-ink-muted)]"
        >
          {content.body}
        </motion.p>

        {content.formula && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.15, duration: 0.7 }}
            className="mt-8 self-start rounded-xl border-l-[3px] border-[var(--color-uni-yellow)] bg-white px-5 py-3 text-xl text-[var(--color-uni-black)] shadow-[0_1px_8px_rgba(0,0,0,0.04)]"
          >
            <Math display>{content.formula}</Math>
          </motion.div>
        )}

        {content.axes && content.axes.length > 0 && (
          <motion.div
            initial="hidden"
            animate="show"
            variants={{
              hidden: {},
              show: { transition: { delayChildren: 1.25, staggerChildren: 0.1 } },
            }}
            className="mt-8 grid grid-cols-1 gap-3 sm:grid-cols-3"
          >
            {content.axes.map((axis) => (
              <motion.div
                key={axis.label}
                variants={{
                  hidden: { opacity: 0, y: 10 },
                  show: {
                    opacity: 1,
                    y: 0,
                    transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] },
                  },
                }}
                className="rounded-xl border border-black/10 bg-white px-4 py-3"
              >
                <div className="flex items-center gap-2">
                  <span
                    className="inline-block h-[7px] w-[7px]"
                    style={{ background: "var(--color-uni-yellow)" }}
                  />
                  <span className="text-sm font-semibold uppercase tracking-[0.18em] text-[var(--color-uni-black)]">
                    {axis.label}
                  </span>
                </div>
                <div className="mt-2 text-[15px] leading-snug text-[var(--color-ink-soft)]">
                  {axis.description}
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </div>

      <FooterStamp />
    </>
  );
}
