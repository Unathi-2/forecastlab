# ForecastLab

An open statistical forecasting platform for analysts, students, and researchers.

Upload a time series, explore its structure, fit forecasting models, compare them with honest validation, and export a report.

## Principles

1. **Reproducibility**: the same data, code, and settings always give the same results. Seeds, versions, and configurations are recorded with every experiment.
2. **Correct validation**: chronological splits and rolling forecasting origin only. No random shuffling of time series.
3. **Statistical transparency**: every model shows its assumptions, diagnostics, and uncertainty.

## Status

Phase 0: scope, datasets, and environment. See `SCOPE.md` and `DATASETS.md`.

## Planned stack

| Layer | Technology |
| --- | --- |
| Frontend | React, TypeScript, Tailwind CSS |
| Backend | Python, FastAPI |
| Statistics | pandas, NumPy, statsmodels, scikit-learn |
| Plots | Plotly |
| Database | PostgreSQL |
| Tests | pytest |
| Automation | GitHub Actions |

## Suggested repository layout

```
forecastlab/
  backend/
    app/
    tests/
  frontend/
  notebooks/
  docs/
  .devcontainer/devcontainer.json
  .github/workflows/ci.yml
  requirements.txt
  README.md
```

## Getting started (Codespaces)

1. Push this repository to GitHub.
2. Open it in a Codespace. The dev container installs Python and dependencies.
3. Run `pytest` to confirm the environment works.