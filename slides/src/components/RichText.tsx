import type { Fragment, RichFragment } from "../content/types";
import { Math } from "./Math";

interface RichTextProps {
  fragments: Fragment[];
}

/**
 * Renderiza una secuencia de fragmentos de texto definidos en el JSON
 * de contenido. Cada fragmento puede aplicar efectos como `highlight`
 * (subrayado amarillo), `italic` o renderizado como fórmula KaTeX.
 */
export function RichText({ fragments }: RichTextProps) {
  return (
    <>
      {fragments.map((fragment, index) => {
        if (typeof fragment === "string") {
          return <span key={index}>{fragment}</span>;
        }
        return <FragmentSpan key={index} fragment={fragment} />;
      })}
    </>
  );
}

function FragmentSpan({ fragment }: { fragment: RichFragment }) {
  if (fragment.math) {
    return <Math>{fragment.text}</Math>;
  }

  const inner = fragment.italic ? <em>{fragment.text}</em> : fragment.text;

  if (fragment.highlight) {
    return <span className="ink-highlight">{inner}</span>;
  }

  return <span>{inner}</span>;
}

/**
 * Renderiza una cadena plana que puede contener fórmulas inline
 * delimitadas por `$...$`. Útil para viñetas o subtítulos cortos.
 *
 *   "Δ = e^{-qT} P_1" → solo texto
 *   "Δ = $e^{-qT} P_1$" → texto + Math con `e^{-qT} P_1`
 */
export function InlineMathText({ children }: { children: string }) {
  const parts = children.split(/(\$[^$]+\$)/g);
  return (
    <>
      {parts.map((part, index) => {
        if (part.startsWith("$") && part.endsWith("$")) {
          return <Math key={index}>{part.slice(1, -1)}</Math>;
        }
        return <span key={index}>{part}</span>;
      })}
    </>
  );
}
