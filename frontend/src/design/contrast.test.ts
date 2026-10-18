// WCAG AA contrast for every text colour on every surface it sits on, in both themes (P8-15).
// Reads the real tokens.css, so a token change that breaks contrast fails CI.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

// Vitest runs from frontend/; CSS imports are stubbed in tests, so read the file itself
const css = readFileSync(resolve(process.cwd(), "src/design/tokens.css"), "utf8");

type Rgb = [number, number, number];
type Theme = Record<string, string>;

function block(selector: string): Theme {
  const start = css.indexOf(selector);
  const body = css.slice(css.indexOf("{", start) + 1, css.indexOf("}", start));
  const tokens: Theme = {};
  for (const [, name, value] of body.matchAll(/(--[\w-]+):\s*([^;]+);/g)) {
    if (name && value) tokens[name] = value.trim();
  }
  return tokens;
}

const light = block(":root {");
const dark = { ...light, ...block(':root[data-theme="dark"]') };

function parse(value: string, theme: Theme): { rgb: Rgb; alpha: number } {
  const ref = /^var\((--[\w-]+)\)$/.exec(value);
  if (ref?.[1]) return parse(theme[ref[1]] ?? "", theme);
  const hex = /^#([0-9a-f]{6})$/i.exec(value);
  if (hex?.[1]) {
    const n = parseInt(hex[1], 16);
    return { rgb: [(n >> 16) & 255, (n >> 8) & 255, n & 255], alpha: 1 };
  }
  const rgb = /^rgb\((\d+) (\d+) (\d+)(?: \/ ([\d.]+))?\)$/.exec(value);
  if (rgb)
    return { rgb: [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])], alpha: Number(rgb[4] ?? 1) };
  throw new Error(`cannot parse colour ${value}`);
}

/** The colour of a token as painted over an opaque base. */
function paint(token: string, theme: Theme, base: Rgb): Rgb {
  const { rgb, alpha } = parse(theme[token] ?? token, theme);
  return rgb.map((c, i) => c * alpha + (base[i] ?? 0) * (1 - alpha)) as Rgb;
}

function luminance([r, g, b]: Rgb): number {
  const ch = (v: number) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b);
}

function contrast(a: Rgb, b: Rgb): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (hi + 0.05) / (lo + 0.05);
}

const page: Record<string, Rgb> = { light: [255, 255, 255], dark: [0, 0, 0] };
const surfaces = [
  "--system-background",
  "--grouped-background",
  "--grouped-cell",
  "--material-bar",
  "--material-sidebar",
];
const text = ["--label", "--secondary-label"];

describe.each([
  ["light", light],
  ["dark", dark],
] as const)("%s theme", (name, theme) => {
  const base = page[name] as Rgb;
  it.each(text.flatMap((t) => surfaces.map((s) => [t, s] as const)))(
    "%s on %s meets AA for body text",
    (fg, bg) => {
      const surface = paint(bg, theme, base);
      expect(contrast(paint(fg, theme, surface), surface)).toBeGreaterThanOrEqual(4.5);
    },
  );

  it("tint text (selected tab) meets AA on the bars", () => {
    for (const bg of ["--material-bar", "--material-sidebar"]) {
      const surface = paint(bg, theme, base);
      expect(contrast(paint("--tint-text", theme, surface), surface)).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("tint text meets AA on the selected sidebar item's highlight", () => {
    const sidebar = paint("--material-sidebar", theme, base);
    const selected = paint("--fill-tertiary", theme, sidebar);
    expect(contrast(paint("--tint-text", theme, selected), selected)).toBeGreaterThanOrEqual(4.5);
  });

  it("caution text meets AA on its badge", () => {
    const cell = paint("--grouped-cell", theme, base);
    const orange = paint("--system-orange", theme, cell);
    const badge = orange.map((c, i) => c * 0.18 + (cell[i] ?? 0) * 0.82) as Rgb;
    expect(contrast(paint("--caution-text", theme, badge), badge)).toBeGreaterThanOrEqual(4.5);
  });

  it("the focus ring and probability bars stand out at 3:1 (non-text)", () => {
    const cell = paint("--grouped-cell", theme, base);
    expect(contrast(paint("--tint", theme, cell), cell)).toBeGreaterThanOrEqual(3);
  });
});
