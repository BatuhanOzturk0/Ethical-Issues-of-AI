# Ethical Issues for AI – HCV-Egy Dataset Fairness Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)


### GISMA University of Applied Sciences  

- **Module:** M515 – Ethical Issues for AI  
- **Student:** Batuhan Öztürk | GH1031500
- **Program:** M.Eng. Computer Science    
- **GitHub:** https://github.com/BatuhanOzturk0/Ethical-Issues-of-AI

---

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Fairness Evaluation](#fairness-evaluation)
- [Results & Model Comparison](#results--model-comparison)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Ethical Considerations](#ethical-considerations)
- [Future Work](#future-work)
- [References](#references)
- [Contributing](#contributing)

---

## 🎯 Overview

This project explores the **ethical implications of using artificial intelligence in healthcare**, specifically focusing on **bias detection and fairness evaluation** in machine learning models that predict Hepatitis C virus (HCV) disease progression stages.

Using the **HCV-Egy dataset** from the UCI Machine Learning Repository, this study builds and evaluates multiple machine learning models while rigorously testing for potential discrimination based on **gender** and **age**—two critical sensitive attributes that could lead to unfair treatment in medical decision-making.

### Why This Matters
AI systems in healthcare can significantly impact patient outcomes. If these systems learn and perpetuate biases from historical data, certain demographic groups may receive:
- Inaccurate diagnoses
- Delayed treatment
- Inappropriate medical interventions
- Unequal access to quality healthcare

This project demonstrates how to identify, measure, and mitigate such biases to build **fair, transparent, and trustworthy AI systems** for medical applications.

---

## 🔍 Problem Statement

The primary goal is to develop a machine learning model that can accurately predict the **stage of liver disease** (Stages 1-4) in Hepatitis C patients based on medical and laboratory features. However, the project goes beyond accuracy to address:

1. **Fairness**: Ensuring predictions are equitable across gender and age groups
2. **Bias Detection**: Identifying potential discriminatory patterns in model predictions
3. **Transparency**: Understanding which features drive predictions and why
4. **Clinical Applicability**: Building models that support fair, data-driven medical decisions

### Research Questions
- Can machine learning accurately predict liver disease stages from medical data?
- Do models exhibit bias toward specific demographic groups (gender, age)?
- Which model architecture provides the best balance between accuracy and fairness?
- How can we quantify and compare fairness across different models?

---

## 📊 Dataset Description

### Source
**HCV-Egy Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)

### Dataset Composition

#### Main File: `HCV-Egy-Data.csv`
- **Total Records**: Multiple patient records from Egypt
- **Target Variable**: `baselinehistological staging` (Disease stages 1-4)
- **Feature Categories**:
  - **Demographic**: Age, Gender, BMI
  - **Liver Enzymes**: ALT (Alanine Transaminase), AST (Aspartate Transaminase)
  - **Blood Tests**: WBC (White Blood Cells), RBC (Red Blood Cells), HGB (Hemoglobin)

#### Key Features

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Patient age in years | Continuous |
| `gender` | Patient gender (1=Male, 2=Female) | Categorical |
| `bmi` | Body Mass Index | Continuous |
| `alt 1` | Alanine Transaminase level | Continuous |
| `ast 1` | Aspartate Transaminase level | Continuous |
| `wbc` | White Blood Cell count | Continuous |
| `rbc` | Red Blood Cell count | Continuous |
| `hgb` | Hemoglobin level | Continuous |
| `baselinehistological staging` | Disease stage (1-4) | Target |

#### Supporting File: `Discretization-Criteria.csv`
Provides categorization thresholds for continuous medical variables, enabling discrete range analysis (e.g., low, normal, high).

### Dataset Characteristics
- **Class Balance**: Nearly balanced across all 4 disease stages (~24-26% each)
- **Gender Distribution**: 51.05% Male, 48.95% Female (well-balanced)
- **Age Range**: Primarily 20-65 years, concentrated in 30-55 age group
- **Data Quality**: Minimal missing values, realistic medical ranges

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)

#### Univariate Analysis
- Distribution analysis of individual features
- Detection of outliers and anomalies
- Assessment of data quality and completeness

**Key Observations**:
- Gender distribution is nearly balanced (~51% male)
- Age distribution is slightly right-skewed
- BMI values fall within normal to slightly overweight ranges (18-30)
- Liver enzyme values (ALT, AST) show expected right-skew with high values in diseased patients
- Blood test values (WBC, RBC, HGB) are within normal medical ranges

#### Bivariate Analysis
**Non-Sensitive Features vs Target**:
- Analyzed relationship between medical features and disease stages
- Created boxplots to visualize feature distributions across stages
- Found that most features show overlapping distributions with minimal stage-specific patterns

**Sensitive Features vs Target**:
- **Gender**: Equal representation across all disease stages
- **Age**: No clear correlation between age and disease progression

#### Multivariate Analysis
- Examined relationships between sensitive and non-sensitive features
- Correlation heatmap revealed weak correlations overall
- No strong feature dominance detected

### 2. Machine Learning Models

Three classification models were implemented and compared:

#### Model Architectures

1. **Support Vector Machine (SVM)**
   ```python
   SVC(gamma='auto', random_state=77)
   ```
   - Kernel-based classification
   - MinMaxScaler preprocessing

2. **Random Forest**
   ```python
   RandomForestClassifier(n_estimators=150, criterion='log_loss', random_state=77)
   ```
   - Ensemble decision tree method
   - 150 trees with log loss criterion

3. **Logistic Regression**
   ```python
   LogisticRegression(penalty='l2', max_iter=1000, random_state=77)
   ```
   - L2 regularization
   - Linear probabilistic classification

#### Training Setup
- **Train-Test Split**: 80-20 stratified split
- **Preprocessing**: MinMaxScaler normalization
- **Random State**: 77 (for reproducibility)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### 3. Fairness Evaluation Framework

Three comprehensive fairness metrics were implemented:

### a) Disparate Impact
**Definition**: Measures whether predictions are distributed equally across demographic groups.

**Method**: Calculates the proportion of each predicted outcome (Stage 1-4) for males vs. females.

**Fairness Criterion**: Similar prediction distributions across genders indicate fairness.

### b) Disparate Mistreatment (Accuracy Gap)
**Definition**: Measures whether model accuracy differs significantly between demographic groups.

**Method**: Compares prediction accuracy for males vs. females.

**Fairness Criterion**: Accuracy difference < 5% is generally considered acceptable.

### c) Disparate Treatment (Error Rate Gap)
**Definition**: Measures whether error rates differ significantly between demographic groups.

**Method**: Calculates proportion of incorrect predictions for each gender.

**Fairness Criterion**: Error rate difference < 5% indicates fair treatment.

---

## 📈 Key Findings

### Exploratory Data Analysis Insights

1. **Dataset Balance**: The target variable is well-balanced across all 4 disease stages, reducing class imbalance issues.

2. **Demographic Fairness**: Both gender and age are fairly distributed across disease stages, suggesting minimal inherent bias in the data.

3. **Feature Relationships**: 
   - Non-sensitive features show weak correlations with the target
   - No single feature dominates disease stage prediction
   - Liver enzymes (ALT, AST) show slight elevation in advanced stages but with high overlap

4. **No Strong Demographic Bias in Features**: Gender and age do not significantly affect medical test results, indicating data-level fairness.

### Model Performance Summary

| Model | Overall Accuracy | Male Accuracy | Female Accuracy | Accuracy Gap |
|-------|-----------------|---------------|-----------------|--------------|
| **SVM** | 28.9% | 25.17% | 34.62% | **9.45%** ⚠️ |
| **Random Forest** | 29.7% | 31.29% | 27.69% | **3.60%** ✅ |
| **Logistic Regression** | 31.3% | 30.61% | 32.31% | **1.70%** ✅ |

### Fairness Analysis Results

#### 1. Disparate Impact Analysis

**SVM - SEVERE BIAS DETECTED** ⚠️
- Predicts Stage 4 for **82.99% of males** but only **16.15% of females** (5x difference)
- Never predicts Stage 2 for females (0.00%)
- Systematically over-diagnoses males with severe disease
- **Verdict**: Unsuitable for clinical deployment

**Random Forest - BEST FAIRNESS** ✅
- Balanced predictions across all stages for both genders
- Stage distribution differences < 7% for all stages
- Most equitable model
- **Verdict**: Recommended for fairness

**Logistic Regression - MODERATE BIAS** ⚠️
- Predicts Stage 4 for 50% of males vs. 25% of females (2x difference)
- Predicts Stage 2 for 33% of males vs. 11% of females (3x difference)
- Shows gender-specific prediction patterns
- **Verdict**: Requires bias mitigation before deployment

#### 2. Disparate Mistreatment (Accuracy Disparity)

**SVM**: 9.45% accuracy gap (unfair to males)
- Males receive significantly worse predictions
- 3 out of 4 male patients are misclassified
- Violates fairness principles

**Random Forest**: 3.60% accuracy gap (acceptable)
- Nearly equal treatment for both genders
- Most balanced model despite low overall accuracy

**Logistic Regression**: 1.70% accuracy gap (excellent fairness)
- Minimal accuracy difference between genders
- Best fairness metric in this category

#### 3. Disparate Treatment (Error Rate Disparity)

Results mirror the accuracy analysis:
- **SVM**: 9.45% error rate gap (discriminatory)
- **Random Forest**: 3.60% error rate gap (fair)
- **Logistic Regression**: 1.70% error rate gap (most fair)

---

## 🏆 Results & Model Comparison

### Comprehensive Model Evaluation

| Metric | SVM | Random Forest | Logistic Regression | Winner |
|--------|-----|---------------|---------------------|--------|
| **Overall Accuracy** | 28.9% | 29.7% | 31.3% | Logistic Regression |
| **Gender Fairness** | Severely Biased | Fair | Most Fair | Logistic Regression |
| **Accuracy Gap** | 9.45% | 3.60% | 1.70% | Logistic Regression |
| **Error Rate Gap** | 9.45% | 3.60% | 1.70% | Logistic Regression |
| **Clinical Suitability** | ❌ Not Suitable | ⚠️ Use with Caution | ✅ Best Option | Logistic Regression |

### Recommended Model: **Logistic Regression**

**Strengths**:
- Highest overall accuracy (31.3%)
- Best fairness metrics across all categories
- Minimal gender bias (1.7% gap)
- Balanced error distribution
- Transparent and interpretable

**Alternative**: **Random Forest** (for ensemble approach)
- Good fairness (3.6% gap)
- Balanced predictions across stages
- Robust to outliers

**Avoid**: **SVM**
- Severe gender bias (9.45% gap)
- Discriminatory predictions
- Systematically harms male patients

---

## 💻 Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/BatuhanOzturk0/Ethical-Issues-of-AI.git
   cd Ethical-Issues-of-AI
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter tqdm
   ```

### Running the Analysis

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**
   - Navigate to `hcv.ipynb` in the Jupyter interface

3. **Run all cells**
   - Execute cells sequentially to reproduce the analysis
   - Cell execution order matters for reproducibility

### Dataset Setup
Ensure the data files are in the correct location:
```
Ethical Issues of AI/
├── hcv.ipynb
├── README.md
└── data/
    ├── HCV-Egy-Data.csv
    └── Discretization-Criteria.csv
```

---

## 📁 Project Structure

```
Ethical-Issues-of-AI/
│
├── README.md                          # This file
├── hcv.ipynb                          # Main analysis notebook
│
├── data/
│   ├── HCV-Egy-Data.csv              # Primary dataset
│   └── Discretization-Criteria.csv   # Feature categorization thresholds
│
└── .venv/                             # Virtual environment (not tracked)
```

### Notebook Sections

1. **Introduction & Problem Statement**
2. **Ethical Discussion**
3. **Dataset Description**
4. **Exploratory Data Analysis**
   - Univariate Analysis
   - Bivariate Analysis (Non-Sensitive Features)
   - Bivariate Analysis (Sensitive Features)
   - Multivariate Analysis
   - Correlation Analysis
5. **Bias Detection**
6. **Machine Learning Models**
   - Model Definition
   - Training & Evaluation
   - Confusion Matrices
7. **Fairness Evaluation**
   - Disparate Impact
   - Disparate Mistreatment
   - Disparate Treatment
8. **Conclusions & Recommendations**
9. **References**

---

## ⚖️ Ethical Considerations

### The Importance of Fairness in Medical AI

Medical AI systems must adhere to the highest ethical standards because their decisions directly impact human health and well-being. This project addresses several critical ethical concerns:

#### 1. **Algorithmic Fairness**
- Ensuring equal treatment regardless of gender, age, or other sensitive attributes
- Preventing systematic discrimination against any demographic group
- Balancing accuracy with equity

#### 2. **Transparency and Explainability**
- Using interpretable models (Logistic Regression) alongside complex ones
- Documenting model behavior comprehensively
- Making bias metrics visible and understandable

#### 3. **Clinical Impact**
- **False Negatives**: Missing severe cases could delay critical treatment
- **False Positives**: Over-diagnosis leads to unnecessary procedures and anxiety
- **Disparate Impact**: Biased predictions harm trust in AI healthcare systems

#### 4. **Data Representation**
- Ensuring diverse patient populations in training data
- Recognizing limitations when applying models to new populations
- Continuous monitoring for emerging biases

### Lessons Learned

1. **High accuracy ≠ Fair model**: SVM had competitive accuracy but severe bias
2. **Balanced data ≠ Bias-free predictions**: Even with balanced training data, models can learn discriminatory patterns
3. **Multiple fairness metrics needed**: Single metrics can miss important disparities
4. **Fairness-accuracy tradeoff**: Sometimes exists, but fairest model (Logistic Regression) was also most accurate here

---

## 🚀 Future Work

### Immediate Improvements

1. **Feature Engineering**
   - Create interaction terms between medical features
   - Engineer domain-specific derived features
   - Apply dimensionality reduction (PCA, feature selection)

2. **Advanced Modeling**
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks with fairness constraints
   - Ensemble methods combining multiple models

3. **Bias Mitigation Techniques**
   - **Pre-processing**: Reweighting, resampling
   - **In-processing**: Fairness-aware training algorithms
   - **Post-processing**: Threshold adjustment per group

### Research Extensions

4. **Additional Fairness Metrics**
   - Equal Opportunity (True Positive Rate parity)
   - Equalized Odds (TPR and FPR parity)
   - Calibration (prediction confidence fairness)

5. **Intersectional Fairness**
   - Analyze combinations (e.g., young females vs. old males)
   - Multi-attribute fairness evaluation

6. **Temporal Validation**
   - Test on data from different time periods
   - Assess model drift and fairness stability

7. **External Validation**
   - Test on datasets from different geographic regions
   - Evaluate cross-population generalizability

8. **Clinical Validation**
   - Collaborate with medical professionals
   - Conduct prospective studies
   - Measure real-world clinical impact

---

## 📚 References

### Dataset
- UCI Machine Learning Repository. (2024). *HCV-Egy Dataset.*  
  [https://archive.ics.uci.edu/datasets](https://archive.ics.uci.edu/datasets)

### Technical Documentation
- Scikit-learn Documentation. [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- Seaborn Documentation. [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
- Pandas Documentation. [https://pandas.pydata.org/](https://pandas.pydata.org/)

### Fairness in Machine Learning
- Mehrabi, N., et al. (2021). "A Survey on Bias and Fairness in Machine Learning." *ACM Computing Surveys*.
- Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning*. fairmlbook.org
- Chouldechova, A., & Roth, A. (2020). "A Snapshot of the Frontiers of Fairness in Machine Learning." *Communications of the ACM*.

### Medical AI Ethics
- Char, D. S., et al. (2020). "Implementing Machine Learning in Health Care—Addressing Ethical Challenges." *New England Journal of Medicine*.
- Rajkomar, A., et al. (2018). "Ensuring Fairness in Machine Learning to Advance Health Equity." *Annals of Internal Medicine*.

---

## 📄 License

This project is available under the MIT License. See the LICENSE file for details.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the HCV-Egy dataset
- The open-source community for excellent ML and data science tools
- Medical professionals working to ensure AI benefits all patients equally

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The models developed here should **NOT be used for actual medical diagnosis or treatment decisions** without:
- Extensive clinical validation
- Regulatory approval
- Supervision by qualified medical professionals
- Proper ethical review and approval

Medical AI systems require rigorous testing, validation, and oversight before clinical deployment.

---

**Built with ❤️ for ethical AI in healthcare**
