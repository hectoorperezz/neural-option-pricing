import type { ReactNode } from "react";
import { CasesSlide } from "./CasesSlide";
import { ClosingSlide } from "./ClosingSlide";
import { ExperimentSlide } from "./ExperimentSlide";
import { ExperimentResultsSlide } from "./ExperimentResultsSlide";
import { ExperimentsOverviewSlide } from "./ExperimentsOverviewSlide";
import { LiteratureSlide } from "./LiteratureSlide";
import { MethodologySlide } from "./MethodologySlide";
import { MotivationSlide } from "./MotivationSlide";
import { QuestionSlide } from "./QuestionSlide";
import { SurrogateSlide } from "./SurrogateSlide";
import { TitleSlide } from "./TitleSlide";
import { TrainingSlide } from "./TrainingSlide";
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
    id: "question",
    label: "Pregunta",
    render: () => <QuestionSlide />,
    notes:
      "La pregunta de investigación: bajo qué condiciones una red puede ser un surrogate preciso, rápido y diferenciable. No predecimos el mercado, sustituimos un solver caro.",
  },
  {
    id: "motivation",
    label: "Motivación",
    render: () => <MotivationSlide />,
    notes:
      "El cuello de botella: los modelos realistas (Heston, Bates) son caros y hay que evaluarlos millones de veces (calibración, Greeks, simulación). Ahí entra el surrogate.",
  },
  {
    id: "literature",
    label: "Revisión bibliográfica",
    render: () => <LiteratureSlide />,
    notes:
      "Cinco referencias núcleo. Chen et al. (deep surrogates) es la ancla; Huge y Savine aportan el differential ML que usamos en E5 (entrenar con derivadas verdaderas).",
  },
  {
    id: "surrogate",
    label: "Deep surrogate",
    render: () => <SurrogateSlide />,
    notes:
      "El esquema: el modelo genera datos sintéticos, la red los aprende offline y luego sustituye al solver. La ventaja: derivadas exactas y baratas por autograd.",
  },
  {
    id: "cases",
    label: "Casos de estudio",
    render: () => <CasesSlide />,
    notes:
      "Black-Scholes como entorno de control (solución cerrada, validamos todo) y Heston como caso real (Fourier semi-cerrado, costoso).",
  },
  {
    id: "methodology",
    label: "Metodología",
    steps: 3,
    render: () => <MethodologySlide />,
    notes:
      "El pipeline común: dominio e hipercubo, muestreo (uniforme vs enfocado), targets y métricas (precio, IV, Delta por autograd) y evaluación por bins 5x5 sobre el test balanceado.",
  },
  {
    id: "experiments-overview",
    label: "Cinco experimentos",
    render: () => <ExperimentsOverviewSlide />,
    notes:
      "Cinco experimentos, cada uno varía una sola dimensión: métrica (E1), activación (E2), muestreo (E3), eficiencia (E4) e información diferencial (E5).",
  },
  {
    id: "training",
    label: "Entrenamiento",
    steps: 3,
    render: () => <TrainingSlide />,
    notes:
      "Antes de los resultados: todo corrió en una estación con i9-14900K, 64 GB de RAM y RTX 4060. Régimen común (MLP 4x128 Swish, Adam, 100 épocas). Y el experimento de escalado: el diseño inicial se estancaba en 1e-2; con el argumento de muestras por parámetro escalamos los datasets x50 (generación de ~10 h, fotos), y el error cayó hasta ~1e-3.",
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
