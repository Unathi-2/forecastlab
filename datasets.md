# Dataset Evaluation Plan

ForecastLab should be tested on several nonfinancial series that differ in frequency, seasonality, and difficulty.

## Selection criteria

- Open licence that allows redistribution or clear download instructions
- At least 100 observations, ideally 200 or more
- Regular time index, or a known reason it is irregular
- Clear seasonality or trend, so models can be told apart
- Some messiness (missing values, outliers) to exercise validation

## Candidate categories

| Category | Typical frequency | What it tests |
| --- | --- | --- |
| Electricity demand | Hourly or daily | Multiple seasonalities |
| Weather (temperature, rainfall) | Daily or monthly | Strong annual seasonality, noise |
| Transport demand | Daily or monthly | Trend plus seasonality, shocks |
| Population or demographic | Annual | Trend only, short series |
| Solar irradiance (for example SAURAN) | Hourly | Strong daily cycle, links to masters research |

## Inspection checklist (per dataset)

- [ ] Source URL and licence recorded
- [ ] Date column parses cleanly
- [ ] Index frequency identified
- [ ] Missing values and duplicates counted
- [ ] STL decomposition run, seasonal period confirmed
- [ ] ADF and KPSS results noted
- [ ] Baseline MASE computed as a reference
- [ ] Decision: include, include as stress test, or reject

## Output

A short table in this file recording the decision for each dataset, plus a `notebooks/` entry showing the inspection.