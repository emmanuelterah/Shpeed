# Shpeed design principles

The UI follows Apple's Human Interface Guidelines. These rules translate them for a data-heavy web app. When in doubt, choose the option a reader understands fastest.

## The principles

**Clarity.** The forecast is the content. Every screen leads with one answer (who is likely to win, how the model has done), in legible type with plenty of space. No decoration that does not carry meaning.

**Deference.** The interface steps back. Chrome is light and translucent, colour is reserved for meaning, and controls appear where they are needed.

**Depth.** Hierarchy comes from layering: translucent bars over scrolling content, grouped lists on a recessed background, sheets that slide over the screen they came from.

**Consistency.** Standard patterns behave the way people expect from Apple platforms: segmented controls switch views of the same data, the large title collapses into the bar as you scroll, back always returns where you were.

## Foundations (all in `src/design/tokens.css`)

| Area          | Rule                                                                                                                                                                                                                                                                                                                                                                          |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Typography    | System font stack (`-apple-system`), which renders SF Pro on Apple devices. Type scale mirrors Dynamic Type: Large Title 34, Title 1 28, Title 2 22, Title 3 20, Headline 17 semibold, Body 17, Callout 16, Subhead 15, Footnote 13, Caption 12/11. Sizes are in `rem` so browser text size settings scale everything. Probabilities use tabular figures.                     |
| Colour        | Semantic tokens modelled on iOS system colours (`--label`, `--secondary-label`, `--separator`, `--system-blue`…), each with a light and a dark value. Never a raw hex in a component. Blue is the tint; green, orange and red only mean good, caution and bad. Text in tint or caution colours uses `--tint-text` and `--caution-text`, darker in light mode so they pass AA. |
| Layout        | 4-point spacing grid. Inset grouped lists for data. Readable width capped near 720 px for text; content respects safe-area insets.                                                                                                                                                                                                                                            |
| Hit targets   | At least 44 × 44 px for anything tappable.                                                                                                                                                                                                                                                                                                                                    |
| Materials     | Navigation and tab bars use a translucent material (`backdrop-filter: saturate(180%) blur(20px)`) with a solid fallback.                                                                                                                                                                                                                                                      |
| Motion        | Short and purposeful (200–350 ms, standard easing). Everything respects `prefers-reduced-motion`.                                                                                                                                                                                                                                                                             |
| Dark mode     | First-class. Follows the system setting; true black background on OLED-style dark.                                                                                                                                                                                                                                                                                            |
| Accessibility | WCAG AA contrast in both themes (checked by `src/design/contrast.test.ts` and an axe scan in the e2e tests), visible focus rings, arrow keys in segmented controls, every chart has a text equivalent, VoiceOver labels on probability bars.                                                                                                                                  |

## Navigation

- **iPhone width:** bottom tab bar with four tabs (Race, Season, Calibration, About) and a large title on each screen.
- **iPad and Mac width:** sidebar navigation on the left, content on the right, the same four destinations.

## Licensing notes

- SF Pro font files and SF Symbols are licensed for Apple platforms only, so they are not bundled. The system font stack gets SF Pro on Apple devices; icons come from Lucide, an open-source set with a similar line weight.
- No official F1 marks or team logos. Teams are identified by name and a neutral colour.
