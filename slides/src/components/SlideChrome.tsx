import { motion } from "framer-motion";

interface SlideChromeProps {
  index: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
  /** Si true, muestra el contador "01 / 12" a la izquierda. En slides
   *  con SideRail visible esto sobra. */
  showCounter?: boolean;
}

/**
 * Cromo inferior con número de slide y controles de navegación discretos.
 * Aparece flotando sobre el contenido pero no interfiere visualmente.
 */
export function SlideChrome({
  index,
  total,
  onPrev,
  onNext,
  showCounter = false,
}: SlideChromeProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, duration: 0.6 }}
      className="pointer-events-none fixed inset-x-0 bottom-0 z-40 flex items-end justify-between px-6 py-5 sm:px-10"
    >
      {showCounter ? (
        <div className="pointer-events-auto flex items-center gap-3 text-xs tracking-widest text-[var(--color-ink-muted)]">
          <span className="font-mono tabular-nums">
            {String(index + 1).padStart(2, "0")}
            <span className="mx-2 text-[var(--color-ink-muted)]/40">/</span>
            {String(total).padStart(2, "0")}
          </span>
        </div>
      ) : (
        <span />
      )}

      <div className="pointer-events-auto flex items-center gap-2">
        <NavButton onClick={onPrev} disabled={index === 0} ariaLabel="Anterior">
          <ArrowIcon direction="left" />
        </NavButton>
        <NavButton onClick={onNext} disabled={index === total - 1} ariaLabel="Siguiente">
          <ArrowIcon direction="right" />
        </NavButton>
      </div>
    </motion.div>
  );
}

interface NavButtonProps {
  onClick: () => void;
  disabled?: boolean;
  ariaLabel: string;
  children: React.ReactNode;
}

function NavButton({ onClick, disabled, ariaLabel, children }: NavButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      className="grid h-9 w-9 place-items-center rounded-full border border-black/15 bg-white/70 backdrop-blur transition hover:border-black/40 hover:bg-[var(--color-uni-yellow)] disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-white/70"
    >
      {children}
    </button>
  );
}

function ArrowIcon({ direction }: { direction: "left" | "right" }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      style={{
        transform: direction === "left" ? "rotate(180deg)" : undefined,
      }}
    >
      <path
        d="M3 7H11M11 7L7 3M11 7L7 11"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
