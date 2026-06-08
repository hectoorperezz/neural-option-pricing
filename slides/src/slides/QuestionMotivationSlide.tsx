import { motion } from "framer-motion";
import { Eyebrow, FooterStamp } from "../components/SlideLayout";
import { RichText } from "../components/RichText";
import { question, motivation } from "../content";

/**
 * Slide combinada: planteamiento de la pregunta de investigación y motivación
 * (más realismo en el modelo implica una evaluación más cara). Van de la mano,
 * la pregunta arriba como hero y debajo el porqué con la escala de coste.
 */
export function QuestionMotivationSlide() {
  return (
    <>
      <Eyebrow>{question.eyebrow}</Eyebrow>

      {/* Pregunta de investigación (hero) */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15, duration: 0.6 }}
        className="mt-1 font-display text-[clamp(1.55rem,3vw,2.7rem)] leading-[1.16] text-[var(--color-uni-black)]"
      >
        {question.lines.map((line) => (
          <span key={line} className="block">
            {line}
          </span>
        ))}
        <span className="block italic">
          <RichText fragments={question.italicLine} />
        </span>
      </motion.div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5, duration: 0.6 }}
        className="mt-4 max-w-[66ch] text-[clamp(0.92rem,1.05vw,1.02rem)] leading-[1.55] text-[var(--color-ink-muted)]"
      >
        {question.body}
      </motion.p>

      {/* Motivación: más realismo, más coste */}
      <div className="mt-7 flex flex-1 flex-col justify-center border-t border-black/10 pt-6">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.6 }}
          className="flex items-start gap-3"
        >
          <span
            className="mt-2 inline-block h-[7px] w-[7px] shrink-0"
            style={{ background: "var(--color-uni-yellow)" }}
          />
          <span className="font-display text-[clamp(1.05rem,1.55vw,1.5rem)] leading-snug text-[var(--color-uni-black)]">
            <RichText fragments={motivation.headline} />
          </span>
        </motion.div>

        {/* Escala de coste por modelo */}
        <motion.div
          initial="hidden"
          animate="show"
          variants={{
            hidden: {},
            show: { transition: { delayChildren: 0.85, staggerChildren: 0.1 } },
          }}
          className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3"
        >
          {motivation.items.map((item) => (
            <motion.div
              key={item.title}
              variants={{
                hidden: { opacity: 0, y: 10 },
                show: {
                  opacity: 1,
                  y: 0,
                  transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] },
                },
              }}
              className={`flex items-center justify-between rounded-2xl border px-6 py-6 ${
                item.accent ? "border-[var(--color-uni-black)]" : "border-black/10"
              }`}
              style={{
                background: item.accent ? "var(--color-uni-yellow)" : "white",
              }}
            >
              <div>
                <div className="eyebrow mb-1.5">{item.note}</div>
                <div className="font-display text-[19px] font-medium leading-tight text-[var(--color-uni-black)]">
                  {item.title}
                </div>
              </div>
              <div className="ml-3 shrink-0 text-right">
                <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--color-ink-muted)]">
                  evaluación
                </div>
                <div className="font-mono text-lg tabular-nums text-[var(--color-uni-black)]">
                  {item.cost}
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Speedup documentado */}
        {motivation.callout && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2, duration: 0.6 }}
            className="mt-6 flex items-start gap-3 rounded-xl border-l-[3px] border-[var(--color-uni-yellow)] bg-white px-5 py-4"
          >
            <div className="max-w-[15rem] shrink-0 text-[12px] font-semibold uppercase tracking-[0.2em] text-[var(--color-ink-muted)]">
              {motivation.callout.kicker}
            </div>
            <div className="text-[15px] leading-snug text-[var(--color-uni-black)]">
              {motivation.callout.body}
            </div>
          </motion.div>
        )}
      </div>

      <FooterStamp />
    </>
  );
}
