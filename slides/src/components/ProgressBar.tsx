import { motion } from "framer-motion";

interface ProgressBarProps {
  index: number;
  total: number;
}

export function ProgressBar({ index, total }: ProgressBarProps) {
  const ratio = total <= 1 ? 1 : (index + 1) / total;

  return (
    <div className="pointer-events-none fixed inset-x-0 top-0 z-50 h-[3px] bg-black/5">
      <motion.div
        className="h-full"
        style={{ backgroundColor: "var(--color-uni-yellow)" }}
        initial={false}
        animate={{ width: `${ratio * 100}%` }}
        transition={{ type: "spring", stiffness: 180, damping: 26 }}
      />
    </div>
  );
}
