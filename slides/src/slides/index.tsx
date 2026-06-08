import type { ReactNode } from "react";
import { ClosingSlide } from "./ClosingSlide";
import { ExperimentSlide } from "./ExperimentSlide";
import { ExperimentResultsSlide } from "./ExperimentResultsSlide";
import { ExperimentsOverviewSlide } from "./ExperimentsOverviewSlide";
import { LiteratureSlide } from "./LiteratureSlide";
import { MethodologyTrainingSlide } from "./MethodologyTrainingSlide";
import { QuestionMotivationSlide } from "./QuestionMotivationSlide";
import { SurrogateCasesSlide } from "./SurrogateCasesSlide";
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
  /** Notas de orador para la vista de presentador (tecla N). */
  notes?: string;
  /** Pasos de revelado dentro de la diapositiva (0 = sin sub-pasos). */
  steps?: number;
}

// Cada experimento genera dos diapositivas: planteamiento y resultados.
const experimentSlides: Slide[] = experiments.flatMap((exp) => [
  {
    id: exp.id.toLowerCase(),
    label: exp.id,
    notes: exp.notes?.setup,
    steps: 2,
    render: () => (
      <ExperimentSlide
        id={exp.id}
        title={exp.title}
        question={exp.question}
        metric={exp.metric}
        surrogates={exp.surrogates}
        hypothesis={exp.hypothesis}
      />
    ),
  },
  {
    id: `${exp.id.toLowerCase()}-res`,
    label: `${exp.id} · Resultados`,
    notes: exp.notes?.results,
    steps: exp.findings.length + 1,
    render: () => (
      <ExperimentResultsSlide
        id={exp.id}
        title={exp.title}
        figure={exp.figure}
        findings={exp.findings}
        conclusion={exp.conclusion}
      />
    ),
  },
]);

export const slides: Slide[] = [
  {
    id: "title",
    label: "Portada",
    bleed: true,
    render: () => <TitleSlide />,
    notes:
      "Presentación del trabajo: redes neuronales como surrogates de funciones de pricing de opciones. Caso de validación Black-Scholes, caso principal Heston.",
  },
  {
    id: "question-motivation",
    label: "Pregunta y motivación",
    render: () => <QuestionMotivationSlide />,
    notes:
      "La pregunta de investigación (¿una red puede ser un surrogate preciso, rápido y diferenciable?) y el porqué: cuanto más realista el modelo, más cara la evaluación, y hay que evaluarlo millones de veces (calibración, Greeks, simulación). No predecimos el mercado, sustituimos un solver caro.",
  },
  {
    id: "literature",
    label: "Revisión bibliográfica",
    render: () => <LiteratureSlide />,
    notes:
      "Cinco referencias núcleo. Chen et al. (deep surrogates) es la ancla; Huge y Savine aportan el differential ML que usamos en E5 (entrenar con derivadas verdaderas).",
  },
  {
    id: "surrogate-cases",
    label: "Deep surrogate y casos",
    render: () => <SurrogateCasesSlide />,
    notes:
      "El esquema: el modelo genera datos sintéticos, la red los aprende offline y luego sustituye al solver (diferenciable por autograd). Y los dos casos: Black-Scholes como validación con solución cerrada y Heston como caso principal, con Fourier semi-cerrado y volatilidad estocástica.",
  },
  {
    id: "methodology-training",
    label: "Metodología y entrenamiento",
    steps: 3,
    render: () => <MethodologyTrainingSlide />,
    notes:
      "El pipeline común: dominio e hipercubo, muestreo (uniforme vs enfocado), targets y métricas (precio, IV, Delta por autograd) y evaluación por bins 5x5 sobre el test balanceado. Y el montaje físico: todo corrió en una estación con i9-14900K, 64 GB y RTX 4060; con el argumento de muestras por parámetro escalamos los datasets x50 y el error cayó de 1e-2 a 1e-3.",
  },
  {
    id: "experiments-overview",
    label: "Seis experimentos",
    render: () => <ExperimentsOverviewSlide />,
    notes:
      "Seis experimentos, cada uno varía una sola dimensión, métrica (E1), activación (E2), muestreo (E3), eficiencia (E4), información diferencial (E5) y profundidad/scheduler (E6).",
  },
  ...experimentSlides,
  {
    id: "closing",
    label: "Cierre",
    bleed: true,
    render: () => <ClosingSlide />,
    notes:
      "Cierre: la métrica importa, la activación gobierna las Greeks, el muestreo enfocado ayuda donde más cuenta, el speedup es de órdenes de magnitud y el differential ML mejora Greeks y eficiencia muestral.",
  },
];
