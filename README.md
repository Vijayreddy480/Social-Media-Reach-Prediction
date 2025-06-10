# 📣 Social Media Reach Prediction

> 🔍 A Machine Learning project to forecast the **engagement reach** of social media posts using historical data, feature engineering, and regression models.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Social%20Media-blueviolet)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🧠 Project Highlights

- 📥 Used structured data from social media YouTube channels
- ⚙️ Feature engineering on views, likes, video length, etc.
- 🧪 Explored multiple regression algorithms:
  - Linear, Ridge, Random Forest, XGBoost
- ✅ **Lasso Regression** selected as final model for best balance of performance and interpretability
- 📈 Evaluated using R², MAE, RMSE
- 📉 Visualized results and feature significance

---

## 🗃️ Dataset: `channel_details.csv`

**Columns:**

| Column            | Description                               |
|-------------------|-------------------------------------------|
| `Views`           | Number of views for the video             |
| `Subscriber Count`| Subscribers of the channel                |
| `Videos count '   | Number of the videos posted               |
| `Publish Date`    | Date video was published                  |
| `Channel Name`    | Name of the content creator               |

---
🧠 Final Model: Lasso Regression
After evaluating several regression algorithms (Linear, Ridge, Random Forest, XGBoost), Lasso Regression was chosen due to its:

.🔍 Feature selection (shrinks irrelevant features)

.📉 Balance between accuracy and model simplicity

.🧠 Stability with multicollinear data

⚠️ Challenges Addressed
.🧹 Missing or inconsistent engagement data

.🔄 Feature scaling for different magnitudes (e.g., likes vs subscribers)

.🧪 Hyperparameter tuning for Lasso (alpha selection)
