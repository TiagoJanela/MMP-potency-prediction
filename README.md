# MMP-potency-prediction
 Potency prediction for Matched Molecular Pairs

Repository for the paper: **Anatomy of Potency Predictions Focusing on Structural Analogues with Increasing Potency Differences Including Activity Cliffs**

## Environment 
Anaconda can be used to install the .yml file provided.

## Notebooks
* **regression_models_mmps.ipynb** - generates regression models with a 50/50% random and stratified splits for MMP datasets. (Run before SHAP notebook)
* **regression_models_shap_mmps.ipynb** - generates SV/SHAP values for SVR and RFR for MMP test sets.
* **data_final_analysis.ipynb** - derives analysis for the computed results, Fig 2, 3 and 4.
* **shap_bit_analysis.ipynb** - derives SV/SHAP analysis, Fig 5.
* **shap_mapping.ipynb** - Map SV/SHAP values to test compounds.

## Folders
* **ML** - contains Python scripts to support model building and data analysis.
* **dataset** - contains the dataset used in the analysis.
* **ccrlib_master** - constains scripts to generate MMP datasets.