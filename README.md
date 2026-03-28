# Road Accident Analysis in Kenya
## Problem

Road traffic accidents are a leading cause of death in Kenya, yet interventions are often not data-driven.

## Objective

To analyze spatial, temporal, and risk factors contributing to road accident severity.

## Dataset
- 2,595 crash records
- Geo-referenced (latitude & longitude)
- Variables: time, road users, severity

## Tools Used
- Python (Pandas, Matplotlib, Seaborn)
- GIS / Geospatial analysis
- Logistic Regression

## Key Insights
- Pedestrians and motorcyclists are the most vulnerable
- Accidents peak during rush hours (morning & evening)
- High-risk hotspots exist in major urban areas
- Fatal crashes can be predicted using statistical models

## Visualizations
- Accident heatmaps
- Time-series plots
- Correlation matrix

## Conclusion

Data-driven approaches can significantly improve road safety by enabling targeted interventions.

# Kenya Road Accident Fatality Predictor

A machine learning web application that predicts the probability of fatal road accidents in Kenya based on road user types and location.

## Features

- 🔮 Predicts fatality probability for road accidents
- 📍 Location-based predictions for major Kenyan cities
- 🚶 Supports different road user types (pedestrians, motorcyclists, matatus)
- 🎯 Real-time predictions with confidence scores
- 🌐 Web interface and REST API

## Model

The model is trained on 2,595 road accident records from Kenya using Logistic Regression with features:
- Road user types (pedestrian, motorcyclist, matatu)
- Geographic location (latitude, longitude)

### Accuracy
- Overall accuracy: 92%
- AUC Score: 0.94

## API Usage

### Predict Endpoint

🔗 Project Link
(https://github.com/lornakendi/road_accidents_proj)
