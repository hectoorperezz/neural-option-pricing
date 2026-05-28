/**
 * Tipos de los JSON de contenido de las diapositivas.
 *
 * La idea: separar el texto del JSX para que cualquiera pueda editar la
 * presentación tocando solo los `.json` de esta carpeta, sin entrar en
 * el código de los componentes.
 *
 * Para textos con énfasis o resaltado tipográfico usamos arrays de
 * fragmentos. Un fragmento puede ser una cadena simple o un objeto con
 * propiedades adicionales:
 *
 *   "hola "
 *   { "text": "mundo", "highlight": true }   ← subrayado amarillo
 *   { "text": "rápida", "italic": true }
 *
 * Los textos con fórmulas matemáticas usan strings LaTeX (compatibles
 * con KaTeX). Ej.: "\\Delta = e^{-qT} P_1".
 */

/** Fragmento de texto inline. Puede ser una cadena simple o un objeto. */
export type Fragment = string | RichFragment;

export interface RichFragment {
  text: string;
  /** Aplica el efecto `ink-highlight` (subrayado amarillo). */
  highlight?: boolean;
  /** Renderiza en cursiva. */
  italic?: boolean;
  /** Renderiza como fórmula KaTeX inline. El texto debe ser LaTeX. */
  math?: boolean;
}

// -------- Title ---------------------------------------------------------

export interface TitleContent {
  subject: string;
  course: string;
  asignaturaLabel: string;
  title: {
    lineOne: string;
    lineTwo: Fragment[];
  };
  subtitle: string;
  authorsLabel: string;
  authors: string[];
}

// -------- Question ------------------------------------------------------

export interface QuestionContent {
  eyebrow: string;
  lines: string[];
  italicLine: Fragment[];
  body: string;
  /** Fórmula formal del problema (renderizada en display). */
  formula?: string;
  /** Ejes que sostienen la pregunta: precisa / rápida / diferenciable. */
  axes?: { label: string; description: string }[];
}

// -------- Motivation ----------------------------------------------------

export interface MotivationContent {
  eyebrow: string;
  headline: Fragment[];
  body: string;
  items: {
    title: string;
    note: string;
    cost: string;
    accent: boolean;
  }[];
  /** Dato destacado al pie (p. ej., el speedup del paper de referencia). */
  callout?: {
    kicker: string;
    body: string;
  };
}

// -------- Literature ----------------------------------------------------

export interface LiteratureContent {
  eyebrow: string;
  headline: Fragment[];
  body: string;
  papers: {
    authors: string;
    year: string;
    title: string;
    contribution: string;
    /** Cómo encaja la referencia en nuestro proyecto. */
    role: string;
    /** Si `true`, se destaca con borde amarillo y badge. */
    central?: boolean;
  }[];
}

// -------- Surrogate -----------------------------------------------------

export interface SurrogateContent {
  eyebrow: string;
  headline: Fragment[];
  body: string;
  steps: {
    id: string;
    title: string;
    /** Subtítulo. Si empieza por `$` el resto se interpreta como LaTeX. */
    subtitle: string;
  }[];
  /** Fórmula del problema, debajo del pipeline. */
  formula?: {
    body: string;
    caption: string;
  };
}

// -------- Cases ---------------------------------------------------------

export interface CasesContent {
  eyebrow: string;
  headline: Fragment[];
  body: string;
  cards: {
    eyebrow: string;
    title: string;
    /** Fórmula resumen renderizada en KaTeX antes de las viñetas. */
    formula?: string;
    /** Lista de viñetas. Strings simples o con LaTeX inline (`$...$`). */
    bullets: string[];
    accent: boolean;
  }[];
}

// -------- ExperimentsOverview -------------------------------------------

export interface ExperimentsOverviewContent {
  eyebrow: string;
  headline: string;
  experiments: {
    id: string;
    title: string;
    summary: string;
    /** Métrica primaria abreviada. LaTeX inline (`$...$`) admitido. */
    metric?: string;
  }[];
}

// -------- Experiment (E1..E5) ------------------------------------------

export interface ExperimentContent {
  id: string;
  title: string;
  question: string;
  /** Métrica primaria. Si contiene `$...$` se renderiza con KaTeX. */
  metric: string;
  surrogates: string[];
  /** Hipótesis previa al experimento. Admite LaTeX inline. */
  hypothesis?: string;
  /** Criterio de clasificación operativa (positivo fuerte / débil / negativo).
   *  Solo aplica a E3 y E5. */
  threshold?: string;
}

// -------- Closing -------------------------------------------------------

export interface ClosingContent {
  eyebrow: string;
  headline: Fragment[];
  /** Lista corta de "key takeaways" opcional. Admite LaTeX inline. */
  takeaways?: string[];
  footer: {
    institution: string;
    subject: string;
  };
}
