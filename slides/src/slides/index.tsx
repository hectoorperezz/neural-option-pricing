import type { ReactNode } from "react";
import { CasesSlide } from "./CasesSlide";
import { ClosingSlide } from "./ClosingSlide";
import { ExperimentSlide } from "./ExperimentSlide";
import { ExperimentsOverviewSlide } from "./ExperimentsOverviewSlide";
import { LiteratureSlide } from "./LiteratureSlide";
import { MotivationSlide } from "./MotivationSlide";
import { QuestionSlide } from "./QuestionSlide";
import { SurrogateSlide } from "./SurrogateSlide";
import { TitleSlide } from "./TitleSlide";
import { experiments } from "../content";

export interface Slide {
  id: string;
  /** Etiqueta corta (para uso interno o futura mini-navegación). */
  label: string;
  /** Render del contenido. La animación de entrada la gestiona SlideStage. */
  render: () => ReactNode;
  /** Si true, el slide ocupa todo el lienzo sin SideRail ni backdrop común
   *  (típico de portada y cierre, que dibujan su propia identidad). */
  bleed?: boolean;
}

const experimentSlides: Slide[] = experiments.map((exp) => ({
  id: exp.id.toLowerCase(),
  label: exp.id,
  render: () => (
    <ExperimentSlide
      id={exp.id}
      title={exp.title}
      question={exp.question}
      metric={exp.metric}
      surrogates={exp.surrogates}
      hypothesis={exp.hypothesis}
      threshold={exp.threshold}
    />
  ),
}));

export const slides: Slide[] = [
  {
    id: "title",
    label: "Portada",
    bleed: true,
    render: () => <TitleSlide />,
  },
  {
    id: "question",
    label: "Pregunta",
    render: () => <QuestionSlide />,
  },
  {
    id: "motivation",
    label: "Motivación",
    render: () => <MotivationSlide />,
  },
  {
    id: "literature",
    label: "Revisión bibliográfica",
    render: () => <LiteratureSlide />,
  },
  {
    id: "surrogate",
    label: "Deep surrogate",
    render: () => <SurrogateSlide />,
  },
  {
    id: "cases",
    label: "Casos de estudio",
    render: () => <CasesSlide />,
  },
  {
    id: "experiments-overview",
    label: "Cinco experimentos",
    render: () => <ExperimentsOverviewSlide />,
  },
  ...experimentSlides,
  {
    id: "closing",
    label: "Cierre",
    bleed: true,
    render: () => <ClosingSlide />,
  },
];
