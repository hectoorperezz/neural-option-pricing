import { Body, Eyebrow, FooterStamp, Headline } from "../components/SlideLayout";
import { InlineMathText } from "../components/RichText";
import { Reveal } from "../components/Reveal";

export interface ExperimentSlideProps {
  id: string;
  title: string;
  question: string;
  /** Métrica primaria. Texto con LaTeX inline (`$...$`). */
  metric: string;
  surrogates: string[];
  /** Hipótesis previa al experimento. */
  hypothesis?: string;
}

export function ExperimentSlide({
  id,
  title,
  question,
  metric,
  surrogates,
  hypothesis,
}: ExperimentSlideProps) {
  return (
    <>
      <Eyebrow>{`Experimento ${id} · Planteamiento`}</Eyebrow>

      <div className="flex flex-1 flex-col justify-center">
        <Headline delay={0.08} size="lg">
          {title}
        </Headline>
        <Body delay={0.18} className="max-w-[80ch] text-[clamp(1.05rem,1.4vw,1.45rem)]">
          <InlineMathText>{question}</InlineMathText>
        </Body>

        <Reveal at={1} className="mt-8 flex flex-wrap items-center gap-3">
          <div className="inline-flex items-center gap-3 rounded-full border border-black/15 px-5 py-2 text-[13px] uppercase tracking-widest text-[var(--color-ink-soft)]">
            <span
              className="inline-block h-2.5 w-2.5"
              style={{ background: "var(--color-uni-yellow)" }}
            />
            Métrica primaria{" "}
            <span className="text-[15px] normal-case tracking-normal text-[var(--color-uni-black)]">
              <InlineMathText>{metric}</InlineMathText>
            </span>
          </div>
          <div className="inline-flex items-center gap-2 rounded-full bg-[var(--color-uni-black)] px-5 py-2 text-[13px] uppercase tracking-widest text-white">
            Surrogates{" "}
            <span className="font-mono text-[14px] normal-case tracking-normal">
              {surrogates.join(" · ")}
            </span>
          </div>
        </Reveal>

        {hypothesis && (
          <Reveal
            at={2}
            className="mt-8 max-w-[84ch] rounded-2xl border-l-4 border-[var(--color-uni-yellow)] bg-white px-7 py-6"
          >
            <div className="text-[12px] font-semibold uppercase tracking-[0.22em] text-[var(--color-ink-muted)]">
              Hipótesis previa
            </div>
            <div className="mt-3 text-[clamp(1.05rem,1.4vw,1.4rem)] leading-relaxed text-[var(--color-uni-black)]">
              <InlineMathText>{hypothesis}</InlineMathText>
            </div>
          </Reveal>
        )}
      </div>

      <FooterStamp />
    </>
  );
}
