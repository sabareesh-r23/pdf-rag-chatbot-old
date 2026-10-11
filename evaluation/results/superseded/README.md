# Superseded evaluation runs (kept for auditability)

These runs were produced by intermediate code versions during the V1.1 fixes and are **not** the
final results. They are kept because they document why later changes were made.

| Run | Code state | Why superseded |
|---|---|---|
| `results_20261011_120043` | V1.1 first cut | Hedged-abstention rule mislabelled M2 (a long uncited partial answer) as an abstention and hid its sources; rule restricted to short hedges in commit "Restrict hedged-abstention rule..." |
| `results_20261011_120459` (main, k=4), `results_20261011_120659` (main, k=8), `results_20261011_121040` (long, k=4), `results_20261011_121152` (long, k=8) | V1.1 (table-number rule 9, no false-premise rule) | A2 false premise produced a fabricated rationale at k=4 (V1 had hedged) and k=8; prompt rule 10 added. Final runs are in `../` |
