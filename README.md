# 🎬 Movie Recommender System

A machine learning-based movie recommendation engine that predicts user ratings for unseen movies and recommends the top 10 movies a user is most likely to enjoy.  
Built using **stacked regression** with Random Forest, Ridge Regression, and GPU-accelerated XGBoost.

---

## 📚 Dataset

We use the [Movies Dataset by Rounak Banik on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset), which combines metadata (genres, vote average) and user ratings.

---

## ⚙️ Project Structure

```
movie-recommender-system/
├── data/             # Raw CSVs (optional to push)
├── models/           # Large .pkl files (ignored via .gitignore)
├── train.py          # Preprocessing + training script
├── recommend.py      # Inference script (top 10 predictions)
├── requirements.txt  # Python dependencies
├── .gitignore        # Prevents tracking large/binary files
└── README.md         # This file
```

---

## 📦 Pretrained Files (Download)

If you're unable to train the model locally, download these files:

| File                | Size   | Description                            | Download Link |
|---------------------|--------|----------------------------------------|---------------|
| `model.pkl`         | ~7 GB  | Trained `StackingRegressor` model      | [Download](https://drive.google.com/your-model-link) |
| `cleaned_data.pkl`  | ~2 GB  | Feature-engineered training data       | [Download](https://drive.google.com/file/d/1L3J_d-7xpmmBdotXRVJlGUkBA3_MYQVN/view?usp=drive_link) |
| `title.pkl`         | <1 MB  | Movie ID-to-title dictionary           | [Download](https://drive.google.com/file/d/1IxBlH3cXTnJ7-bNt1YyJXF74aQQImIbj/view?usp=drive_link) |

📁 Place all `.pkl` files in the **project root** or inside the `models/` folder.

---

## 🛠️ Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Shrestha-Kumar/stacked-movie-recommender-ml.git
cd movie-recommender-system

# 2. Install required libraries
pip install -r requirements.txt

# 3. 🚀 Train the model (optional)
# Skip this if you already downloaded model.pkl
python train.py
```

This trains a `StackingRegressor` (Random Forest + Ridge + XGBoost) and saves the output files.

---

## 🎯 Generate Movie Recommendations

```bash
python recommend.py --userId 10
```

This script predicts ratings for all unseen movies by the specified user and prints the **Top 10 movie titles** with the highest predicted scores.

---

## 📈 Model Performance

| Metric       | Value  |
|--------------|--------|
| **R² Score** | 0.4037 |
| **RMSE**     | 0.8240 |

> The model explains 40% of user rating variance — solid for a hybrid content-based + collaborative filter.

---

## 🧠 Techniques Used

- ✅ One-hot encoding for genres
- ✅ Feature engineering (mean rating by user/movie, rating deviation, etc.)
- ✅ Ensemble learning using `StackingRegressor`
- ✅ GPU-enabled `XGBoost` to accelerate training
- ✅ `Joblib` for efficient model serialization
- ✅ Final recommendation = `predict → sort → pick top 10`

---

## 🙋‍♂️ Author

**Shresth Kumar**  
B.Tech Electrical Engineering, 2nd Year  
Machine Learning & Data Science Enthusiast

---

## 📜 License

This project is for educational and personal use only.  
Dataset © Kaggle / Rounak Banik.