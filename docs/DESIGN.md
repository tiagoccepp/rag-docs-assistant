# UI design contract — RAG Docs Assistant ("Evidence")

Family: dark ops console shared across the portfolio (Sora display / Public Sans body /
JetBrains Mono data, self-hosted woff2 inlined by `scripts/inline_fonts.py`).
Edit `src/static/ui.src.html`, regenerate with `python scripts/inline_fonts.py`.

## Palette (OKLCH-tuned hex)

- `--bg: #0B0F14` (page) · `--panel: #10161D` · `--raised: #151D26`
- `--line: rgba(233,237,242,.09)` (hairlines) · `--text: #E8EDF2` · `--muted: #93A1B0`
- `--accent: #E5A13D` (amber — knowledge/highlighter; the ONE accent)
- Status: answered emerald `#34C77B`, abstained violet `#9B7EDE`, error `#E5484D`
- Contrast: body text ≈13:1, muted ≈7:1 on panel (both ≥4.5 ✓)

## Layout concept

"Analyst copilot": conversation left, evidence rail right. The answer streams
token-by-token; every `[n]` citation is a button that spotlights its passage.

```text
+--------------------------------------------------------------+
| ◈ RAG Docs Assistant        mock LLM · 26 chunks · API docs  |
+------------------------------+-------------------------------+
| SAMPLE CARDS (4)             |  EVIDENCE RAIL                |
| [user msg →]                 |  #1 Working Hours  ▓▓▓ 0.885 |
| [← assistant streams…]       |  #2 Remote Work    ▓▓  0.767  |
| [ask input____________] [Ask] |  stats: model · latency       |
+------------------------------+-------------------------------+
```

## Bold element

Citations-as-spotlights: hovering/clicking `[1]` pulses the matching passage
card in the rail. Everything else stays quiet and disciplined.

## Motion (all user-answered; `prefers-reduced-motion` respected)

- Load: header → cards → input stagger 60ms each, translateY 8px + fade
- Ask: typing dots in assistant bubble; rail cards shimmer-skeleton while fetching
- Answer: token streaming (SSE), citation buttons pop in at end
- Abstain: violet panel slide-in, rail dims (nothing to cite — honest empty state)

## States

idle (hint) · searching · answered · abstained · error (API down → message + retry).
Mobile ≤900px: rail stacks below conversation; 390px no horizontal overflow.
