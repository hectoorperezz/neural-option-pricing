import { SideRail } from "./components/SideRail";
import { SlideBackdrop } from "./components/SlideBackdrop";
import { SlideChrome } from "./components/SlideChrome";
import { SlideStage } from "./components/SlideStage";
import { useSlideNavigation } from "./hooks/useSlideNavigation";
import { slides } from "./slides";

export function App() {
  const { index, direction, total, goNext, goPrev } = useSlideNavigation(slides.length);
  const current = slides[index];
  const bleed = current.bleed ?? false;

  return (
    <div
      className="relative h-screen w-screen"
      style={{ background: "var(--color-page)" }}
    >
      {!bleed && <SlideBackdrop />}
      {!bleed && <SideRail index={index} total={total} />}

      <SlideStage index={index} direction={direction} bleed={bleed}>
        {current.render()}
      </SlideStage>

      <SlideChrome
        index={index}
        total={total}
        onPrev={goPrev}
        onNext={goNext}
        showCounter={bleed}
      />
    </div>
  );
}
