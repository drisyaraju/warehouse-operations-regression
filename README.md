# 🏭 WarehouseIQ — Order Processing Time Predictor

> *How long will it take? Let the data decide.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Powered-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Statistical%20Analysis-4B8BBE?style=for-the-badge)](https://www.statsmodels.org)
[![Status](https://img.shields.io/badge/Status-Complete-2ECC71?style=for-the-badge)]()

---

## 🔍 What's This About?

Every second counts in a warehouse. A delayed order means a frustrated customer.

**WarehouseIQ** uses regression analysis and machine learning to **predict order processing time** — giving warehouse managers the foresight to optimize operations, allocate staff, and slash bottlenecks before they happen.

---

## 🧠 Models Benchmarked

| Model | Type | Strengths |
|---|---|---|
| 📈 Simple Linear Regression | Baseline | Interpretable, fast |
| 📊 Multiple Linear Regression | Statistical | Multi-feature relationships |
| 🌀 Polynomial Regression | Non-linear | Captures curved patterns |
| 🌳 Decision Tree | Tree-based | Handles non-linearity, visual |
| 🌲 **Random Forest** | Ensemble ⭐ | **Best performer** |
| ⚡ Support Vector Machine | Kernel-based | Robust on high-dimensional data |

> 🏆 **Random Forest** emerged as the top model — outperforming simpler regressors by capturing complex, non-linear interactions in warehouse data.

---

## 📐 Statistical Rigour

This project goes beyond just fitting models — it validates them:

- **Overall F-test** → Is the model as a whole statistically significant?
- **Partial F-test** → Which individual predictors actually matter?

Because a model that *looks* good but isn't statistically sound is just noise.

---

## ⚙️ Project Workflow

```
Raw Data
   │
   ▼
🧹 Preprocessing        — Handle missing values, encode categoricals, scale features
   │
   ▼
🔬 Feature Selection     — Identify the predictors that drive processing time
   │
   ▼
✂️  Train-Test Split     — Fair evaluation on unseen data
   │
   ▼
🤖 Model Training        — 6 models trained and tuned
   │
   ▼
📊 Evaluation            — R², RMSE, MAE compared across all models
   │
   ▼
✅ Best Model Selected   — Random Forest 🌲
```

---

## 📊 Evaluation Metrics

| Metric | What It Tells Us |
|---|---|
| **R² Score** | How much variance in processing time the model explains |
| **RMSE** | How far off predictions are, penalizing large errors |
| **MAE** | The average absolute prediction error in real units |

---

## 🛠️ Tech Stack

```python
import pandas as pd          # Data wrangling
import numpy as np           # Numerical computing
from sklearn import ...      # ML models & evaluation
import statsmodels.api as sm # Statistical testing
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/drisyaraju/warehouseiq.git
cd warehouseiq

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python main.py
```

---

## 📁 Project Structure

```
warehouseiq/
├── data/
│   └── warehouse_orders.csv     # Dataset
├── notebooks/
│   └── analysis.ipynb           # Full walkthrough
├── models/
│   └── random_forest_model.pkl  # Saved best model
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

---

## 💡 Key Takeaways

- **Tree-based ensemble methods** significantly outperform linear approaches on warehouse data — the relationships between features are inherently non-linear.
- **Feature selection** was critical: not all warehouse variables contribute meaningfully to processing time.
- **Statistical testing** confirmed that the top model's predictors weren't just noise — they had genuine explanatory power.

---

## 👩‍💻 Author

**Drisya Raju**

*Built with curiosity, Python, and a deep respect for supply chain efficiency.*

[![GitHub](https://img.shields.io/badge/GitHub-DrisyaRaju-181717?style=flat-square&logo=github)](https://github.com/drisyaraju)

---

<p align="center">
  <i>⭐ If this project helped you, consider giving it a star!</i>
</p>
