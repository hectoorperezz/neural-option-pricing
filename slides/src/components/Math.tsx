import { useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

interface MathProps {
  /** Expresión LaTeX. Usa exactamente la misma sintaxis que en el paper. */
  children: string;
  /** Si es true, renderiza en modo `display` (fórmula centrada en bloque
   *  con tamaño más grande). Por defecto es inline. */
  display?: boolean;
  /** Color del texto. Por defecto hereda del contenedor. */
  color?: string;
}

/**
 * Renderiza una expresión LaTeX con KaTeX. Permite que las fórmulas del
 * paper se reutilicen aquí sin reescribirlas con caracteres Unicode.
 *
 * Ejemplos:
 *   <Math>{"\\Delta = e^{-qT} P_1"}</Math>
 *   <Math display>{"C = S e^{-qT} P_1 - K e^{-rT} P_2"}</Math>
 */
export function Math({ children, display = false, color }: MathProps) {
  const html = useMemo(
    () =>
      katex.renderToString(children, {
        displayMode: display,
        throwOnError: false,
        strict: "ignore",
        output: "html",
      }),
    [children, display],
  );

  if (display) {
    return (
      <span
        style={color ? { color } : undefined}
        className="my-2 block text-center"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  return (
    <span
      style={color ? { color } : undefined}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
