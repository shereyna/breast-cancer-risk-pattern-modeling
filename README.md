# Breast Cancer Outcome Prediction Using Machine Learning Models on BCSC Mammography Data

**GRAD 50800 Final Project Report**  
Author: Shereyna Shinbo
Purdue University | Fall 2025 

## Project Overview
This project evaluates the predictive performance of Logistic Regression and Random Forest models in identifying short-term breast cancer outcomes using screening data from the Breast Cancer Surveillance Consortium (BCSC). The objective is to classify whether a cancer diagnosis occurs within one year of a screening mammogram.

The dataset consists of approximately 20,000 digital and 20,000 film mammography examinations collected between 2005 and 2008. All predictors are derived from radiologist assessments and patient characteristics recorded at the time of screening; no imaging files are included.

Logistic Regression achieved:
- **Accuracy:** 0.9947  
- **AUC:** 0.9109  

Random Forest achieved:
- **Accuracy:** 0.9940  
- **AUC:** 0.8876  

Both models performed strongly, with Logistic Regression demonstrating slightly better discrimination.

## Research Questions
### 1. How accurately can supervised learning models predict one-year breast cancer outcomes?
**Tested Models:**
- Logistic Regression  
- Random Forest  

**Results:**
- Logistic Regression: Accuracy = 0.9947, AUC = 0.9109  
- Random Forest: Accuracy = 0.9940, AUC = 0.8876  

**Conclusion:**  
Both models perform strongly, with Logistic Regression demonstrating slightly higher discrimination.

### 2. Do the models identify similar predictors as most influential?
**Logistic Regression Findings:**
- Strongest positive predictors:
  - Biopsy history  
  - Family history  
- BI-RADS assessment showed a strong negative coefficient in this instructional dataset.

**Random Forest Findings:**
- Highest feature importance:
  - BMI  
  - Age  
  - BI-RADS assessment  

**Conclusion:**  
Predictor rankings differ due to model structure. Logistic Regression evaluates linear log-odds effects, while Random Forest ranks features by impurity reduction across tree splits.

## Methodology
All analyses were conducted using Python with the following framework:

### Data Preparation
- Converted missing-value codes (“9” and “-99”) into formal missing values  
- Removed incomplete or invalid records  
- Dropped Patient ID (ptid) to prevent dependence  
- Dropped Cancer Type (CaTypeO) to prevent outcome leakage  
- Reduced dataset from ~40,000 records to ~15,000 usable examinations  
- Performed 80/20 train–test split with fixed random seed  

### Modeling
**Logistic Regression**
- Penalized maximum likelihood estimation  
- Increased maximum iterations to 1000  
- Generated predicted probabilities and class labels  
- Converted coefficients to odds ratios for interpretation  

**Random Forest**
- 500 decision trees  
- Bootstrap aggregation  
- No depth restriction  
- Majority voting for classification  
- Extracted feature importance scores  

### Evaluation Metrics
- Accuracy  
- Area Under the ROC Curve (AUC)  

## Results Summary
| Research Question | Model/Test | Metric | Result | Interpretation |
|------------------|------------|--------|--------|---------------|
| RQ1: Predictive Accuracy | Logistic Regression | AUC | 0.9109 | Strong discrimination |
| RQ1: Predictive Accuracy | Random Forest | AUC | 0.8876 | Slightly lower discrimination |
| RQ2: Predictor Influence | Coefficients / Feature Importance | — | Model-dependent rankings | Structural differences across models |

Overall, Logistic Regression provided slightly better discrimination and clearer interpretability, while Random Forest captured nonlinear relationships without improving performance.

## Discussion
Logistic Regression captured dominant clinical risk patterns, particularly biopsy history and family history. Random Forest emphasized continuous predictors such as BMI and age due to tree-based splitting mechanics.

The minimal performance gap suggests limited nonlinear structure in the dataset. The contrast between models demonstrates how algorithmic structure influences variable importance interpretation.

## Limitations
- Dataset developed for instructional use, limiting generalizability  
- No imaging files included  
- Significant sample reduction after preprocessing  
- Single train–test split without external validation  

## Full Report
The complete written analysis, including figures, statistical outputs, and interpretation, is available in PDF format: 
[View Full Report](report/GRAD50800_Final Report.pdf)

## Repository Structure

```
evaluating-usda-food-price-outlook/
├── data/
│   └── dataset.csv
│
├── report/
│   ├── graphs/     # Exported visualizations
│   └── GRAD50800_Final Report.pdf
│
├── script/
│   └── CourseProject.py             # Main analysis pipeline and statistical testing
│
└── README.md             # Project overview
```

### Folder Descriptions
- **data/** – Contains dataset used in analysis
- **report/** – Includes the final written report and exported graphs.  
- **script/** – Contains the main Python analysis script for data processing and visualization.   
- **README.md** – Provides an overview of the project and documentation.
