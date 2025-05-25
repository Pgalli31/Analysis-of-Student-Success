# Student Success Analysis in Higher Education

This project explores the factors that influence academic performance in higher education using clustering and regression techniques. The study is based on a dataset of university students and aims to discover meaningful patterns and predictors of student success.

## 🎯 Objective

To analyze and interpret student performance data using:
- **K-Prototypes Clustering** for mixed data types
- **Multiple Linear Regression** for identifying significant predictors

The project seeks to uncover characteristics such as personal background, family support, and study habits that may correlate with academic achievement.

## 📊 Dataset

- Source: UCI Machine Learning Repository
- Total Records: 145 students
- Features: Personal, Family, Educational attributes (e.g., number of siblings, reading habits, scholarship type, study hours)
- Target: Final grade

## 🧪 Methods

### 1. **Data Preprocessing**
- Replaced numerical categorical codes with descriptive labels
- Binned final grades into ranges for clustering analysis
- Converted ordinal features and created dummy variables for regression

### 2. **K-Prototypes Clustering**
- Used to group students based on categorical + numerical data
- Clusters analyzed by grade distribution to identify success patterns
- Subsets:
  - **Personal Attributes**: age, partner status, accommodation
  - **Family Background**: parents’ education, number of siblings
  - **Academic Behavior**: reading habits, attendance, study approach

### 3. **Multiple Linear Regression**
- Modeled final grade as a continuous outcome
- Removed non-significant features iteratively
- Assessed multicollinearity and assumption violations

## 📈 Key Findings

- **Clustering** showed potential correlations:
  - Students with **partners** or **many siblings** often performed better
- **Regression analysis** found only one significant positive factor:
  - Reading **non-academic material** often had a small but meaningful correlation with higher grades
- Several expected academic behaviors (e.g., frequent studying or attendance) were not statistically significant

## ⚠️ Limitations

- The dataset was relatively small and imbalanced
- Final grade may not fully represent academic success
- Model assumptions (homoscedasticity, independence) were violated
- Self-reported data may contain inaccuracies

## 📁 Files

- `student_success_analysis.ipynb` – Full analysis notebook
- `cluster_plots/` – Cluster visualizations by student attribute category
- `regression_model_summary.txt` – Output summary of regression model

## 📘 Future Improvements

- Use **Random Forest or XGBoost** to assess feature importance
- Explore **graduation status** as a success metric instead of grade
- Use larger or more diverse datasets to improve generalizability

---

**Author:** Paul Galli  
