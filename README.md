# Predicting Forest Carbon Stocks from Climate & CO2 Emissions Data

## Overview

This project investigates whether national-level climate data and CO2 emissions
can predict forest carbon stocks (above- and below-ground biomass) across
countries. It combines three public datasets into a single pipeline built with
PySpark, then benchmarks Linear Regression, Random Forest, and Gradient
Boosted Trees for the prediction task.

**Short answer: not very well** — and the reasons why turned out to be more
interesting than the models themselves. See [Results](#results) and
[Why the Models Underperformed](#why-the-models-underperformed) below.

## Data Sources

| Dataset | Source | Content |
|---|---|---|
| FAO FRA 2020 | [fra-data.fao.org](https://fra-data.fao.org/assessments/fra/2020/WO/data-download/) | Carbon in above-ground biomass (AGB) and below-ground biomass (BGB), by country, 1990–2020 |
| NOAA GSOY | [NOAA NCEI](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00947/html) | Station-level annual climate summaries (temperature, precipitation, wind, degree-days) |
| Our World in Data | [ourworldindata.org](https://ourworldindata.org/co2-and-greenhouse-gas-emissions) | CO2 emissions per capita, by country and year |

## Pipeline

1. **FAO FRA preprocessing** — parsed multi-header Excel sheets, dropped
   redundant columns, reshaped into a clean country × year carbon table for
   both AGB and BGB.
2. **NOAA GSOY preprocessing** — matched individual weather stations to
   countries via nearest-centroid geospatial lookup (no country field exists
   in the raw station data), aggregated station-year records into
   country-year climate averages, then bucketed into the five FAO target
   years (1990, 2000, 2010, 2015, 2020). Missing values were imputed first
   with the country mean, then the global mean as a fallback.
3. **OWID CO2 preprocessing** — filtered out aggregate/region rows (e.g.
   "World", "Europe", income-group labels), pivoted to one column per target
   year.
4. **Join** — merged all three sources on `country` and `year`, summed AGB +
   BGB into a single `carbon_total` target.
5. **Modeling** — feature set: average temperature, average precipitation,
   average wind, and CO2 emissions per capita. 70/30 train-test split.

All intermediate and final tables are cached to HDFS as Parquet.

## Results

| Model | R² | RMSE |
|---|---|---|
| Linear Regression | 0.091 | 39.86 |
| Random Forest | **0.240** | **36.45** |
| Gradient Boosted Trees | 0.101 | 39.65 |

Random Forest was the best performer, roughly 2.5x the baseline R², and
identified CO2 per capita as the most important feature (~39% of split
importance), followed by wind and temperature. K-Means clustering + PCA were
used to explore structure in the feature space; clusters loosely separated
countries by climate/emissions profile but didn't map cleanly onto carbon
stock levels.

## Why the Models Underperformed

An R² of 0.24 means the models explain less than a quarter of the variance in
forest carbon stocks. A few likely reasons:

- **Feature-target mismatch.** National climate averages and per-capita CO2
  emissions are economic/atmospheric signals, not direct drivers of forest
  biomass. What actually determines a country's forest carbon — deforestation
  rate, forest age and species composition, soil type, land-use policy — isn't
  represented in the feature set at all.
- **Spatial resolution mismatch.** Forest carbon is highly local (it varies
  by region within a country), while every feature here is a single national
  average. This washes out the local signal FAO's own data is measuring.
- **Correlation, not mechanism.** CO2 per capita showing up as the top
  feature is plausible but suspect — it's more likely a proxy for
  industrialization or economic development that happens to correlate with
  certain forest profiles, not a causal driver of biomass carbon.
- **Small, noisy sample.** ~190 countries × 5 years, with real heterogeneity
  in forest types (boreal, tropical, temperate) lumped into one global model.

## Next Steps

- Incorporate satellite-derived vegetation indices (e.g. NDVI) or
  higher-resolution land-cover data instead of country-level averages.
- Add deforestation/land-use-change features, which are more directly tied
  to biomass carbon than climate or emissions.
- Model forest biome types separately rather than pooling all countries into
  one global regression.

## Tech Stack

PySpark, HDFS, pandas, scikit-learn-style Spark ML pipelines
(`VectorAssembler`, `StandardScaler`, `LinearRegression`,
`RandomForestRegressor`, `GBTRegressor`, `KMeans`, `PCA`), seaborn/matplotlib
for EDA and visualization.
