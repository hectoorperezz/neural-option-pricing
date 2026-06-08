import { motion } from "framer-motion";
import type { ReactNode } from "react";

interface EyebrowProps {
  children: ReactNode;
  delay?: number;
}

export function Eyebrow({ children, delay = 0 }: EyebrowProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="eyebrow mb-4"
    >
      <span className="inline-flex items-center gap-2">
        <span
          aria-hidden
          className="inline-block h-[6px] w-[6px] bg-[var(--color-uni-yellow)]"
        />
        {children}
      </span>
    </motion.div>
  );
}

interface HeadlineProps {
  children: ReactNode;
  delay?: number;
  size?: "xl" | "lg" | "md";
  font?: "serif" | "sans";
}

export function Headline({
  children,
  delay = 0.05,
  size = "lg",
  font = "serif",
}: HeadlineProps) {
  const sizeClass =
    size === "xl"
      ? "text-[clamp(2.6rem,6vw,5.4rem)] leading-[1.02]"
      : size === "lg"
      ? "text-[clamp(2rem,4.5vw,3.8rem)] leading-[1.06]"
      : "text-[clamp(1.4rem,3vw,2.2rem)] leading-[1.15]";

  const fontClass = font === "serif" ? "font-display" : "font-sans";

  return (
    <motion.h1
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      className={`${fontClass} ${sizeClass} font-medium text-[var(--color-uni-black)]`}
    >
      {children}
    </motion.h1>
  );
}

interface BodyProps {
  children: ReactNode;
  delay?: number;
  className?: string;
}

export function Body({ children, delay = 0.18, className = "" }: BodyProps) {
  return (
    <motion.p
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      className={`mt-5 max-w-[68ch] text-[clamp(1rem,1.25vw,1.2rem)] leading-[1.6] text-[var(--color-ink-soft)] ${className}`}
    >
      {children}
    </motion.p>
  );
}

interface StaggerProps {
  children: ReactNode;
  className?: string;
  delay?: number;
  stagger?: number;
}

export function Stagger({
  children,
  className = "",
  delay = 0.25,
  stagger = 0.08,
}: StaggerProps) {
  return (
    <motion.div
      initial="hidden"
      animate="show"
      variants={{
        hidden: {},
        show: {
          transition: {
            staggerChildren: stagger,
            delayChildren: delay,
          },
        },
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

export const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] },
  },
};

interface FooterStampProps {
  label?: string;
}

export function FooterStamp({ label = "Métodos numéricos y simulación estocástica · 2025/2026" }: FooterStampProps) {
  return (
    <div className="mt-auto flex items-end justify-between pt-12">
      <div className="flex items-center gap-3 text-xs tracking-[0.18em] text-[var(--color-ink-muted)] uppercase">
        <img src="/h-mono.png" alt="" className="h-4 w-auto opacity-70" />
        <span>{label}</span>
      </div>
    </div>
  );
}
