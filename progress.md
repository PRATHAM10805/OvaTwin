# OvaTwin Progress

## Overall Project Status

**Core computational framework implemented and fully set up to run.**

OvaTwin contains working components for:

- PCOS machine-learning prediction (XGBoost)
- Random Forest model comparison
- SHAP explainability
- 541-patient clinical data analysis and validation
- Clinical quality flags & preprocessing
- HPO observation construction (`HPOPatientValues`)
- Röblitz 2013 physiological model integration (SBML/OMEX loaded via Tellurium/RoadRunner)
- Baseline simulation (0–120 days, 1201 points)
- Patient digital-twin framework (`PatientDigitalTwin`)
- Patient mapping (`PatientTwinMapping`)
- What-if scenario simulation (`ScenarioSimulator`)
- Baseline-vs-scenario comparison
- Integration and validation test suite

The major remaining research component is:

**validated patient-specific physiological calibration.**

---

# 1. Repository Status

## Status: OPERATIONAL & TESTED

Current source structure:

```text
src/
├── Clinical/
│   ├── clinical_analysis.py
│   ├── clinical_cleaning.py
│   ├── clinical_loader.py
│   ├── clinical_preprocessing.py
│   ├── clinical_quality_check.py
│   └── hpo_observation_builder.py
├── HPO/
│   ├── baseline_simulation.py
│   ├── gyncycle_loader.py
│   ├── model_runner.py
│   ├── parameters.py
│   ├── patient_values.py
│   ├── test_baseline_simulation.py
│   ├── test_model_runner.py
│   └── validation.py
├── integration/
│   ├── test_all_hpo_observations.py
│   ├── test_digital_twin.py
│   ├── test_hpo_observations.py
│   └── test_parameter_mapping.py
├── ML/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── feature_selection.py
│   ├── model_comparison.py
│   ├── preprocessing.py
│   └── train.py
└── twin/
    ├── digital_twin.py
    ├── parameter_mapping.py
    ├── personalization.py
    └── scenarios.py
```

## Setup & Verification

- **Environment**: Python 3.13+, Tellurium, libRoadRunner, python-libsbml, SHAP, Imbalanced-Learn, XGBoost, Scikit-Learn.
- **Models**:
  - `models/HPO/MODEL1311220000.3.omex` downloaded and verified from BioModels.
  - `models/BIOMD0000000494.xml` extracted for GynCycle model loader.
  - `models/xgboost_pcos_model.pkl` trained and saved.
- **Datasets**:
  - `data/PCOS_data_without_infertility-Full_new.csv` (541 patients) configured and validated.
  - `data/clinical_dataset_preprocessed.csv` generated.
  - `data/clinical_dataset_flagged.csv` generated.
  - `data/dataset.csv` configured for the ML prediction pipeline.
- **Tests**:
  - `python main.py` -> Runs ML pipeline (Train, Eval, Comparison, SHAP).
  - `python -m src.integration.test_digital_twin` -> Baseline & scenario simulation passes.
  - `python -m src.integration.test_all_hpo_observations` -> All 541 patients pass validation.
  - `python -m src.integration.test_parameter_mapping` -> Parameter mapping verified.
  - `python -m src.HPO.test_model_runner` -> Röblitz model runner verified.
  - `python -m src.HPO.test_baseline_simulation` -> 120-day baseline simulation verified.