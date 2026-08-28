# Crowd Density Prediction and Congestion Analysis

## 📌 Project Overview

This project develops a machine learning system for predicting crowd
density at different locations inside a large-scale venue.

The system combines **seat-level sensor data, event metadata, and crowd
movement data** to predict `People_Count`. The resulting predictions are
exported to CSV and visualized in a **Power BI dashboard** for spatial
crowd monitoring and congestion analysis.

The complete workflow is:

``` text
Multiple CSV Files
        ↓
Data Integration
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Feature Scaling
        ↓
Multiple Regression Models
        ↓
Model Comparison
        ↓
Ridge Hyperparameter Tuning
        ↓
Final Test Evaluation
        ↓
Prediction Export
        ↓
Power BI Dashboard
```

------------------------------------------------------------------------

# 🎯 Problem Statement

Large-scale venues can experience unpredictable crowd movement, crowd
accumulation, and congestion around specific zones and pathways.

Event organizers and security teams need a predictive system that can:

-   Estimate crowd density at different locations.
-   Identify areas with high predicted crowd levels.
-   Analyze crowd movement through inflow and outflow.
-   Support gate-level and hotspot-level analysis.
-   Provide an interactive visualization for operational monitoring.

The primary machine learning target is:

``` text
People_Count
```

------------------------------------------------------------------------

# 🗂️ Dataset

The project combines three datasets.

## 1. Seat Clusters

File:

``` text
seat_clusters.csv
```

Important columns:

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `People_Count`                      Target variable representing the
                                      observed people count

  `Turnstile_Count`                   Number of people passing through
                                      gate scanners

  `BLE_Pings`                         Bluetooth Low Energy device
                                      detections

  `Pressure_Mat_Activations`          Floor pressure sensor activations

  `Detected_Heads`                    Computer vision detected head count

  `Seat_X`                            X-coordinate of the
                                      seating/location point

  `Seat_Y`                            Y-coordinate of the
                                      seating/location point

  `Seat_Occupancy_Prob`               Estimated seat occupancy
                                      probability

  `Zone_Capacity`                     Maximum capacity of the zone

  `Seat_ID`                           Unique seat/location identifier

  `Frame_ID`                          Camera frame identifier

  `Visibility_Score`                  Camera visibility quality score
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 2. Event Metadata

File:

``` text
event_metadata.csv
```

Important columns:

  Column                 Description
  ---------------------- ---------------------------------
  `Event_ID`             Unique event identifier
  `Total_Attendance`     Total attendance
  `Zone_Attendance`      Attendance for a specific zone
  `Sellout_Rate(%)`      Percentage of tickets sold
  `Evacuation_Time`      Estimated venue evacuation time
  `Staff_On_Duty`        Number of active security staff
  `Timestamp`            Event timestamp
  `Gate_ID`              Entry gate identifier
  `Ticket_Class`         Ticket category
  `Purchase_Type`        Ticket purchase method
  `Gate_Status`          Current gate status
  `Alerts`               Active security alerts
  `Drill_Timestamp`      Last safety drill timestamp
  `Congestion_Hotspot`   Labeled congestion hotspot

------------------------------------------------------------------------

## 3. Movement Edges

File:

``` text
movement_edges.csv
```

Important columns:

  Column               Description
  -------------------- ----------------------------
  `Source_Seat`        Starting location
  `Target_Seat`        Destination location
  `Current_Flow`       Current people flow
  `Flow_Capacity`      Maximum safe path capacity
  `Path_Type`          Physical pathway type
  `Distance`           Physical path distance
  `Congestion_Level`   Derived congestion metric

------------------------------------------------------------------------

# 🔗 Data Integration

The datasets are integrated using common identifiers.

### Event-level join

``` text
Event_ID
```

is used to connect seat-level data with event metadata.

### Seat-level movement join

``` text
Event_ID + Seat_ID
```

is used to attach calculated inflow and outflow information.

The movement dataset is aggregated before merging.

### Inflow

Inflow is calculated by grouping movement records by:

``` text
Event_ID
Target_Seat
```

and summing:

``` text
Current_Flow
Flow_Capacity
```

The resulting columns are renamed to:

``` text
Seat_ID
Inflow
Inflow_Capacity
```

### Outflow

Outflow is calculated by grouping movement records by:

``` text
Event_ID
Source_Seat
```

and summing:

``` text
Current_Flow
Flow_Capacity
```

The resulting columns are renamed to:

``` text
Seat_ID
Outflow
Outflow_Capacity
```

Missing movement values are filled with zero where appropriate.

------------------------------------------------------------------------

# 🔎 Exploratory Data Analysis

The EDA stage examines:

1.  Target distribution.
2.  Sensor-to-target relationships.
3.  Spatial crowd distribution.
4.  Feature correlations.

## Target Distribution

A histogram with KDE is used to inspect the distribution of:

``` text
People_Count
```

This helps understand the range, spread, and potential skewness of the
target variable.

## Sensor Relationship

A scatter plot compares:

``` text
Detected_Heads
```

against:

``` text
People_Count
```

The project data shows a very strong relationship between the
camera-based head count and the target.

## Spatial Analysis

The coordinates:

``` text
Seat_X
Seat_Y
```

are plotted to visualize the spatial distribution of observations.

The point color represents crowd intensity.

## Correlation Analysis

A numerical correlation matrix is generated to identify variables with
strong linear relationships with:

``` text
People_Count
```

------------------------------------------------------------------------

# 🛠️ Feature Engineering and Preprocessing

The selected model features are:

``` python
features = [
    'Turnstile_Count',
    'BLE_Pings',
    'Pressure_Mat_Activations',
    'Detected_Heads',
    'Visibility_Score',
    'Seat_X',
    'Seat_Y',
    'Total_Attendance',
    'Zone_Attendance',
    'Inflow',
    'Outflow'
]
```

Target:

``` python
target = 'People_Count'
```

The input matrix is:

``` python
X = df_raw[features]
```

and the target vector is:

``` python
y = df_raw[target]
```

------------------------------------------------------------------------

# ✂️ Train / Test Split

The dataset is divided into:

``` text
80% → Training data
20% → Testing data
```

using:

``` python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The test dataset remains unseen during model training and hyperparameter
selection.

------------------------------------------------------------------------

# 📏 Feature Scaling

`StandardScaler` is used to standardize numerical features.

The standardization formula is:

\[ z = `\frac{x-\mu}{\sigma}`{=tex} \]

where:

-   `x` = original value
-   `μ` = training-data mean
-   `σ` = training-data standard deviation
-   `z` = standardized value

The scaler is fitted only on the training data:

``` python
X_train_scaled = preprocessor.fit_transform(X_train)
```

The same learned transformation is applied to the test data:

``` python
X_test_scaled = preprocessor.transform(X_test)
```

This prevents the test set from influencing the preprocessing
parameters.

------------------------------------------------------------------------

# 🤖 Model Building and Comparison

Five regression algorithms are evaluated:

``` text
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor
```

## Evaluation Metrics

### RMSE

Root Mean Squared Error measures the typical magnitude of prediction
error.

\[ RMSE = `\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}`{=tex} \]

Lower RMSE indicates better predictive accuracy.

### R²

R² measures the proportion of target variance explained by the model.

\[ R\^2 = 1 - `\frac{
\sum(y_i-\hat{y}_i)^2
}{
\sum(y_i-\bar{y})^2
}`{=tex} \]

A value closer to `1.0` indicates a stronger fit.

------------------------------------------------------------------------

# 📊 Baseline Model Results

The model comparison produced approximately:

  Model                   RMSE       R²
  ------------------- -------- --------
  Linear Regression     1.4404   0.9989
  Decision Tree         2.7350   0.9960
  Random Forest         1.8749   0.9981
  Gradient Boosting     1.7437   0.9984
  XGBoost               1.7506   0.9984

Based on the lowest RMSE, **Linear Regression** was selected as the best
baseline model.

------------------------------------------------------------------------

# 🎛️ Cross Validation and Fine Tuning

Because Linear Regression was the best baseline model, a regularized
linear extension was evaluated using **Ridge Regression**.

Ridge Regression adds L2 regularization to the linear regression
objective:

\[ `\min`{=tex}\_{`\beta`{=tex}} `\left[
\sum_i(y_i-\hat{y}_i)^2
+
\alpha\sum_j\beta_j^2
\right]`{=tex}\]

The hyperparameter:

``` text
alpha
```

controls the strength of regularization.

The tested values were:

``` python
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}
```

Grid Search uses:

``` text
5-fold cross-validation
```

to evaluate the candidate values.

The scoring metric is:

``` text
neg_root_mean_squared_error
```

Scikit-learn uses the negative form because GridSearchCV treats higher
scores as better.

The best configuration found was:

``` text
alpha = 0.01
```

------------------------------------------------------------------------

# 🧪 Final Test Evaluation

The tuned model is evaluated on the previously unseen test set.

Approximate final results:

``` text
Final RMSE = 1.4398
Final R²   = 0.9989
```

The model's actual-vs-predicted plot shows predictions lying very close
to the ideal:

``` text
Actual = Predicted
```

reference line.

------------------------------------------------------------------------

# 📈 Actual vs Predicted Visualization

The final evaluation creates a scatter plot where:

``` text
X-axis → Actual People Count
Y-axis → Predicted People Count
```

The diagonal reference line represents perfect predictions.

Points close to this line indicate small prediction errors.

------------------------------------------------------------------------

# 💾 Artifact Export

The trained preprocessing object is exported as:

``` text
data_preprocessor.pkl
```

The trained final model is exported as:

``` text
tuned_crowd_model.pkl
```

Predictions for the test dataset are exported as:

``` text
powerbi_final_predictions.csv
```

The prediction CSV contains the original test observations plus:

``` text
Predicted_People_Count
```

------------------------------------------------------------------------

# 📊 Power BI Dashboard

The exported prediction data is loaded into Power BI.

The dashboard provides an operational view of predicted crowd density.

## Main Components

### Peak Predicted Crowd Density

Displays the maximum predicted crowd count.

Example:

``` text
198.34
```

------------------------------------------------------------------------

### Predicted Crowd Density Map

The main spatial visualization uses:

``` text
Seat_X
Seat_Y
```

as the coordinate system.

Each bubble represents a location.

Bubble size and color provide visual information about predicted crowd
levels.

Higher predicted density is highlighted using the dashboard's intensity
scale.

------------------------------------------------------------------------

### Gate Filter

The dashboard provides a slicer for:

``` text
East
North
South
West
```

Selecting a gate filters the dashboard to the corresponding
observations.

------------------------------------------------------------------------

### Congestion Hotspot Filter

The dashboard also provides a slicer for hotspot zones such as:

``` text
Z1
Z2
Z3
Z6
Z7
Z9
Z10
```

This allows users to investigate predicted density associated with
specific congestion areas.

------------------------------------------------------------------------

# 🧱 Project Architecture

``` text
                 ┌───────────────────────┐
                 │   Seat Clusters CSV   │
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │  Event Metadata CSV   │
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │  Movement Edges CSV   │
                 └───────────┬───────────┘
                             │
                             ▼
                   Data Integration
                             │
                             ▼
                           EDA
                             │
                             ▼
                   Feature Engineering
                             │
                             ▼
                    Train / Test Split
                             │
                             ▼
                     Feature Scaling
                             │
                             ▼
                    Model Comparison
                             │
             ┌───────────────┴──────────────┐
             │                              │
             ▼                              ▼
      Linear Regression              Tree/Boosting Models
             │
             ▼
       Ridge Regression
             │
             ▼
       GridSearchCV
             │
             ▼
       Final Prediction
             │
             ▼
      CSV Prediction Export
             │
             ▼
          Power BI
```

------------------------------------------------------------------------

# 📁 Recommended Project Structure

``` text
Crowd_Optimization/
│
├── Dataset/
│   └── archive/
│       ├── event_metadata.csv
│       ├── movement_edges.csv
│       └── seat_clusters.csv
│
├── Notebook/
│   └── crowd_congestion.ipynb
│
├── data_preprocessor.pkl
├── tuned_crowd_model.pkl
├── powerbi_final_predictions.csv
│
├── PowerBI/
│   └── crowd_congestion.pbix
│
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# ⚙️ Installation

## 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Crowd_Optimization
```

## 2. Create a virtual environment

``` bash
python -m venv venv
```

### Windows

``` bash
venv\Scripts\activate
```

### Linux / macOS

``` bash
source venv/bin/activate
```

## 3. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 📦 Main Dependencies

The project uses:

``` text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

------------------------------------------------------------------------

# ▶️ How to Run

## Step 1 --- Prepare the datasets

Place the three CSV files in:

``` text
Dataset/archive/
```

------------------------------------------------------------------------

## Step 2 --- Open the notebook

Open:

``` text
crowd_congestion.ipynb
```

using Jupyter Notebook, JupyterLab, or VS Code.

------------------------------------------------------------------------

## Step 3 --- Update dataset paths

If required, change:

``` python
meta = pd.read_csv(...)
edges = pd.read_csv(...)
seats = pd.read_csv(...)
```

to match your local dataset location.

------------------------------------------------------------------------

## Step 4 --- Run the notebook

Run the sections in order:

``` text
1. Problem Statement
2. Data Dictionary
3. Data Loading and Integration
4. Exploratory Data Analysis
5. Feature Engineering and Preprocessing
6. Model Building and Comparison
7. Cross Validation and Fine Tuning
8. Prediction and Test Evaluation
9. Artifact Export
```

------------------------------------------------------------------------

## Step 5 --- Verify generated files

After successful execution, you should have:

``` text
data_preprocessor.pkl
tuned_crowd_model.pkl
powerbi_final_predictions.csv
```

------------------------------------------------------------------------

# 🔮 Using the Saved Model for New Predictions

For new raw observations, the preprocessing object must be applied
before the model.

Conceptually:

``` python
import joblib

preprocessor = joblib.load("data_preprocessor.pkl")
model = joblib.load("tuned_crowd_model.pkl")

new_data_scaled = preprocessor.transform(new_data)

predictions = model.predict(new_data_scaled)
```

The new data must contain the same feature columns used during training:

``` text
Turnstile_Count
BLE_Pings
Pressure_Mat_Activations
Detected_Heads
Visibility_Score
Seat_X
Seat_Y
Total_Attendance
Zone_Attendance
Inflow
Outflow
```

------------------------------------------------------------------------

# ⚠️ Important Model Validation Note

The EDA shows extremely strong correlations between some sensor
variables and `People_Count`.

In particular, values close to:

``` text
Detected_Heads       → People_Count ≈ 1.00
Turnstile_Count      → People_Count ≈ 1.00
```

should be investigated before treating the model performance as
production-ready.

If any input feature was directly calculated from `People_Count`, the
model would suffer from **target leakage**, causing artificially high
performance.

A proper audit should verify:

``` text
Input feature
      ↓
Was it available before prediction?
      ↓
Was it independently measured?
      ↓
Was it derived from People_Count?
      ↓
No target leakage?
```

This validation is especially important because the reported performance
is extremely high:

``` text
RMSE ≈ 1.44
R² ≈ 0.999
```

------------------------------------------------------------------------

# 🔐 Reproducibility

The project uses:

``` python
random_state=42
```

for deterministic train/test splitting and supported model
initialization.

This makes experiments easier to reproduce when the same dataset and
software environment are used.

------------------------------------------------------------------------

# 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates:

-   Data integration
-   Data cleaning
-   GroupBy aggregation
-   Feature engineering
-   Train/test splitting
-   Standardization
-   Regression
-   Model comparison
-   RMSE
-   R²
-   Cross-validation
-   Grid Search
-   Hyperparameter tuning
-   L2 regularization
-   Model serialization
-   Prediction export
-   Business intelligence visualization

------------------------------------------------------------------------

# 📌 Final Results

``` text
Best baseline model:
Linear Regression

Best Ridge hyperparameter:
alpha = 0.01

Final RMSE:
≈ 1.4398

Final R²:
≈ 0.9989
```

These results should be interpreted together with the target-leakage
audit described above.

------------------------------------------------------------------------

# 📊 Business Value

The system can support venue operations by helping teams:

-   Identify locations with high predicted crowd density.
-   Monitor potential congestion areas.
-   Compare crowd conditions across gates.
-   Analyze spatial crowd distribution.
-   Improve security resource allocation.
-   Support crowd-flow planning.
-   Provide a visual monitoring interface through Power BI.

------------------------------------------------------------------------

# 🚀 Future Improvements

Potential improvements include:

1.  Perform a formal target-leakage audit.
2.  Use time-aware validation if observations are temporal.
3.  Compare against a leakage-free feature set.
4.  Add MAE and MAPE as additional evaluation metrics.
5.  Use SHAP or permutation importance for model interpretability.
6.  Add prediction intervals/uncertainty estimates.
7.  Incorporate temporal features such as hour, event phase, and rolling
    flow.
8.  Add automated model retraining.
9.  Deploy the model through an API.
10. Connect Power BI to a continuously updated prediction source.
11. Add automated congestion alerts for high-risk zones.

------------------------------------------------------------------------

# 👨‍💻 Project Summary

This project demonstrates an end-to-end machine learning workflow for
**crowd density prediction and congestion analysis**.

The system integrates heterogeneous venue data, performs exploratory
analysis and preprocessing, compares multiple regression algorithms,
applies cross-validated Ridge regularization, evaluates the final model
on unseen data, exports reusable ML artifacts, and presents the
predictions through an interactive Power BI dashboard.

The result is a complete pipeline from:

``` text
Raw Data
   ↓
Machine Learning
   ↓
Prediction
   ↓
Business Intelligence
```

------------------------------------------------------------------------

