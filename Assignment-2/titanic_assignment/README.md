# Titanic Survival Prediction Assignment

This project focuses on predicting Titanic survival rates through an end-to-end data pipeline, marking a transition from basic exploratory data analysis to a structured machine learning pipeline.

## Structure
The project is built around modularization:
- `data/`: Contains the raw and processed datasets (e.g., `train.csv`).
- `notebooks/`: Contains `Titanic_Feature_Engineering.ipynb`, showcasing exploratory cleaning, feature creation, and visualization.
- `scripts/`: Python scripts extracting the analytical logic into modular steps:
  - `data_cleaning.py`: Imputes missing values and removes outliers.
  - `feature_engineering.py`: Engineers derived features like `FamilySize` and encodes variables.
  - `feature_selection.py`: Ranks the important features using Random Forest.

## How to Run
1. Place `train.csv` and `test.csv` in the `data/` directory.
2. Install requirements using `pip install -r requirements.txt`.
3. Run the notebook `notebooks/Titanic_Feature_Engineering.ipynb` to step through the entire analysis.
4. Alternatively, run the scripts in sequence:
   - `python scripts/data_cleaning.py`
   - `python scripts/feature_engineering.py`
   - `python scripts/feature_selection.py`

## Data Cleaning Summary
- **Age and Fare**: Missing values were imputed using the median.
- **Cabin**: Dropped due to a high rate of missingness; instead, tracked as an indicator feature `HasCabin`.
- **Embarked**: Filled missing values with the mode.
- **Fare**: Capped top outliers at the 99th percentile to prevent skewing.
- Duplicates were identified and removed.

## Engineered Features
- **FamilySize**: Combines siblings/spouses and parents/children aboard.
- **IsAlone**: Boolean indicator if traveling without family.
- **Title**: Extracted from the `Name` field (Mr, Mrs, Miss, Rare, etc.).
- **FarePerPerson**: Average fare per family member.
- **LogFare**: A logarithmic transformation of `Fare` to normalize its distribution.
- **Categorical Columns**: Sex, Embarked, and Title were one-hot encoded.

## Feature Selection Findings
After running a Random Forest Classifier and inspecting feature importances: 
- `Sex` (female/male) remains the most critical deterministic feature.
- `Fare` (and its Log/Per-person transformations) and `Age` carry significant weight.
- Derived features like `Title` provide meaningful groupings.
- Lower ranking features include `IsAlone` or certain embarkation ports, which provided less predictive power overall.
