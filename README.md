# Live Venue Crowd Density and Congestion Optimization Pipeline

## Project Overview
Large-scale venues and stadiums frequently suffer from unpredictable crowd flows and corridor bottlenecks. This capstone project delivers an end-to-end machine learning and business intelligence solution designed to anticipate crowd surges at specific seating coordinates, empowering security teams to allocate resources dynamically and optimize turnstile throughput.

---

## Repository Structure
```text
├── archive/
│   ├── event_metadata.csv      # Event context, attendance, and gate logs
│   ├── movement_edges.csv      # Foot traffic paths, flow capacities, and source/target seats
│   └── seat_clusters.csv       # Sensor readings, spatial coordinates, and zone capacities
├── notebooks/
│   └── dynamic_crowd.ipynb     # Complete data engineering, EDA, and model tuning notebook
├── data_preprocessor.pkl       # Serialized scikit-learn standard scaling preprocessor artifact
├── tuned_crowd_model.pkl       # Serialized hyperparameter-tuned regression model artifact
├── powerbi_final_predictions.csv# Scored dataset imported into Power BI
├── crowd_congestion.pbix       # Interactive Power BI dashboard report
├── enviroment.yaml             # Conda environment configuration file
└── README.md                   # Project documentation