import { AnimatePresence, motion } from "framer-motion";
import type { Variants } from "framer-motion";
import type { Direction } from "../hooks/useSlideNavigation";
import type { ReactNode } from "react";

interface SlideStageProps {
  index: number;
  direction: Direction;
  bleed?: boolean;
  children: ReactNode;
}

const variants: Variants = {
  enter: (d: Direction) => ({
    opacity: 0,
    x: d * 32,
    filter: "blur(6px)",
  }),
  center: {
    opacity: 1,
    x: 0,
    filter: "blur(0px)",
    transition: {
      duration: 0.55,
      ease: [0.22, 1, 0.36, 1] as [number, number, number, number],
    },
  },
  exit: (d: Direction) => ({
    opacity: 0,
    x: -d * 32,
    filter: "blur(6px)",
    transition: {
      duration: 0.35,
      ease: [0.4, 0, 0.6, 1] as [number, number, number, number],
    },
  }),
};

export function SlideStage({ index, direction, bleed = false, children }: SlideStageProps) {
  return (
    <div className="relative h-full w-full overflow-hidden">
      <AnimatePresence mode="wait" custom={direction} initial={false}>
        <motion.div
          key={index}
          className={bleed ? "slide-frame-bleed" : "slide-frame"}
          custom={direction}
          variants={variants}
          initial="enter"
          animate="center"
          exit="exit"
        >
          {children}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
