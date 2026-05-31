/**
 * Punto de entrada del contenido. Importa cada JSON y lo tipa con el
 * tipo correspondiente. Las diapositivas consumen estos exports en
 * lugar de los `.json` directamente, para que TypeScript valide los
 * campos y sea fácil renombrar o añadir nuevos.
 */
import titleData from "./title.json";
import questionData from "./question.json";
import motivationData from "./motivation.json";
import literatureData from "./literature.json";
import surrogateData from "./surrogate.json";
import casesData from "./cases.json";
import methodologyData from "./methodology.json";
import trainingData from "./training.json";
import experimentsOverviewData from "./experiments-overview.json";
import experimentsData from "./experiments.json";
import closingData from "./closing.json";

import type {
  CasesContent,
  ClosingContent,
  ExperimentContent,
  ExperimentsOverviewContent,
  LiteratureContent,
  MethodologyContent,
  MotivationContent,
  TrainingContent,
  QuestionContent,
  SurrogateContent,
  TitleContent,
} from "./types";

export const title = titleData as TitleContent;
export const question = questionData as QuestionContent;
export const motivation = motivationData as MotivationContent;
export const literature = literatureData as LiteratureContent;
export const surrogate = surrogateData as SurrogateContent;
export const cases = casesData as CasesContent;
export const methodology = methodologyData as MethodologyContent;
export const training = trainingData as TrainingContent;
export const experimentsOverview =
  experimentsOverviewData as ExperimentsOverviewContent;
export const experiments = experimentsData as ExperimentContent[];
export const closing = closingData as ClosingContent;
