# 💻 LaptiQ
### Know what your laptop is worth

LaptiQ is an end-to-end machine learning web app that predicts the current Indian market price of any laptop based on its specifications. Enter your CPU, GPU, RAM, display, and other details — LaptiQ returns an estimated price range with explainability.

**🚀 Live Demo →** https://laptiq.streamlit.app/

---

## ✨ What it does

- Takes raw laptop specs as input
- Engineers features automatically (PPI, CPU tier, GPU tier, Laptop Age)
- Runs everything through a trained XGBoost pipeline
- Returns a predicted price **range** in Indian Rupees (₹) with ±12% margin
- Shows the **top 5 factors** that drove the prediction using SHAP explainability
- Lets you **compare up to 3 laptops** side-by-side with a Best Value badge
- Keeps a session history of past predictions with delete support

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| R² Accuracy | 93.46% |
| Mean Absolute Error | 11.68% |
| Adjusted R² | 93.31% |
| Training samples | 935 laptops |

---

## 🗂️ Project Structure

```
laptiQ/
│
├── data/
│   ├── Laptop_Prices.csv               # raw dataset — 937 laptops, 18 features
│   ├── Laptop_Prices_Featured.csv      # after feature engineering — 935 rows, 20 cols
│   └── Laptop_Prices_Model_Ready.csv   # fully encoded & ready for model — 935 rows, 33 cols
│
├── model/
│   └── LaptiQ.pkl                      # trained pipeline saved with joblib
│
├── notebooks/
│   ├── Exploratory_Data_Analysis.ipynb # full EDA — distributions, correlations, feature decisions
│   └── Model.ipynb                     # model benchmarking — 7 algorithms compared
│
├── src/
│   ├── __init__.py
│   └── preprocess.py                   # feature engineering function used by train.py and app.py
│
├── app.py                              # Streamlit web app
├── train.py                            # pipeline training + GridSearchCV
├── requirements.txt
└── README.md
```

---

## 🗃️ Data Files Explained

Three versions of the dataset are included so you can follow the full transformation journey:

**`Laptop_Prices.csv`** — Raw data as collected. 937 laptops, 18 columns. String columns like `CPU_Model`, `GPU_Model`, `Resolution` exactly as captured.

**`Laptop_Prices_Featured.csv`** — After `preprocess()` runs. Raw strings replaced with engineered features:

```
+ Laptop_Age       (from Launch_Year)
+ CPU_Series       (low / mid / high — from CPU_Model)
+ CPU_Segment      (low / mid / high — from CPU_Model suffix)
+ CPU_Generation   (modern / latest — from CPU_Model)
+ GPU_Tier         (low / mid / high — from GPU_Model)
+ Pixel_Per_Inch   (from Resolution + Screen_Size)

- Launch_Year, CPU_Model, GPU_Model, Resolution, Screen_Size, Storage_Type removed
```

**`Laptop_Prices_Model_Ready.csv`** — Fully encoded and scaled. What the XGBoost model actually trains on — 33 numeric columns, no strings. This is what the model sees and learns from.

---

## ⚙️ How It Works

**Step 1 — EDA**

Analysed 937 laptops across 18 features. Key decisions made:
- Price is right-skewed (mean ₹1.44L vs median ₹1.08L) → log transform on target
- RAM (0.79) and Storage (0.79) are strongest correlators with Price
- `Storage_Type` dropped — 99% SSD, near-zero variance
- `Screen_Size` + `Resolution` → PPI — more informative than either alone
- `CPU_Model` (138 unique values) → 3 ordinal features instead of one-hot encoding
- `GPU_Model` (52 unique values) → GPU_Tier (low/mid/high)

**Step 2 — Feature Engineering**

| Raw Column | Engineered Output | Logic |
|---|---|---|
| `Launch_Year` | `Laptop_Age` | 2026 − Launch_Year |
| `CPU_Model` | `CPU_Series` | regex → low/mid/high tier |
| `CPU_Model` | `CPU_Segment` | suffix (H/HX/U/etc.) → low/mid/high |
| `CPU_Model` | `CPU_Generation` | modern vs latest chip architecture |
| `GPU_Model` | `GPU_Tier` | regex → low/mid/high tier |
| `Resolution` + `Screen_Size` | `Pixel_Per_Inch` | √(W²+H²) / screen_size |
| `GPU_VRAM` | `GPU_VRAM` (int) | "Shared"→0, strip "GB" → int |
| `Storage_Type` | — | dropped (99% SSD) |

**Step 3 — Pipeline**

Full sklearn `Pipeline` with `ColumnTransformer` — all transformers fit on train data only, no leakage:

```
ColumnTransformer
├── OrdinalEncoder       → GPU_Tier, CPU_Series, CPU_Segment, CPU_Generation
├── OneHotEncoder        → Brand, Laptop_Type, CPU_Brand, GPU_Type, OS
├── FunctionTransformer  → log1p on RAM, Storage
├── PowerTransformer     → Yeo-Johnson on Weight, Pixel_Per_Inch
└── RobustScaler         → CPU_Cores, GPU_VRAM, Laptop_Age

TransformedTargetRegressor  → log1p(Price) during fit, expm1 at predict
└── XGBRegressor (tuned via GridSearchCV — 108 combinations × 5 folds)
```

Best params: `n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8`

**Step 4 — Price Range**

Instead of a single predicted value, LaptiQ returns a ±12% price range rounded to the nearest ₹500 — honest about the model's real uncertainty (MAE is 11.68%) without false precision.

```
Predicted:  ₹1,27,855
Range:      ₹1,12,500 – ₹1,43,000
```

**Step 5 — SHAP Explainability**

After every prediction, LaptiQ uses SHAP (SHapley Additive exPlanations) to show the top 5 factors that influenced the price — with approximate rupee impact for each.

One-hot encoded columns (Brand, Laptop Type, OS etc.) are grouped back into their parent feature before ranking, so the UI shows "Laptop Type" not 6 separate binary columns.

```
↑ Display Quality (PPI)        +₹33,973
↑ Graphics Tier                +₹30,935
↓ Storage Capacity             -₹13,154
↓ RAM                          -₹10,579
↓ Graphics Memory (VRAM)        -₹4,899
```

**Step 6 — App**

Streamlit UI with four pages:
- 🏠 **Home** — enter specs, get price range + SHAP factors
- ⚖️ **Compare** — compare up to 3 laptops side-by-side, Best Value badge on the cheapest
- 🕒 **History** — view and delete past predictions (session-based, max 20)
- 📊 **Model Info** — model stats and how it works

---

## 🛠️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Yash-Codezz/LaptiQ.git
cd LaptiQ
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser. Works on mobile too via your local network URL.

**To retrain the model from scratch:**
```bash
python train.py
```
Takes ~1 minute. Saves the new model to `model/LaptiQ.pkl`.

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Model | XGBoost |
| ML Pipeline | scikit-learn — Pipeline, ColumnTransformer, TransformedTargetRegressor, GridSearchCV |
| Preprocessing | OrdinalEncoder, OneHotEncoder, FunctionTransformer, PowerTransformer, RobustScaler |
| Explainability | SHAP — TreeExplainer with grouped feature attribution |
| UI | Streamlit |
| Data | pandas, numpy |
| Serialisation | joblib |

---

## 🧠 Key Decisions

**Why XGBoost?**
Benchmarked 7 models — Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost. XGBoost had the best R² (92.2% pre-tuning, 93.46% post-tuning) and lowest MAE across the board.

**Why log transform on Price?**
Price is heavily right-skewed. Log transform brings it closer to normal and improves model performance significantly. `TransformedTargetRegressor` handles this inside the pipeline — no manual inverse transform needed.

**Why PPI instead of Resolution + Screen Size separately?**
Both had weak individual correlations with price. Combined as pixel density, the signal is much stronger. A 4K 13" screen is very different from a 4K 17" screen — PPI captures that.

**Why drop Storage_Type?**
99% of laptops in the dataset are SSD. The column carries almost no information.

**Why 3 CPU features instead of one-hot encoding CPU_Model?**
`CPU_Model` had 138 unique values. One-hot encoding would create 138 sparse columns. Breaking it into `CPU_Series` + `CPU_Segment` + `CPU_Generation` captures the meaningful variance in 3 clean ordinal columns.

**Why a price range instead of a single value?**
A single number implies false precision. The model's MAE is 11.68% — on a ₹1.2L laptop that's ±₹14K. Showing a range is more honest and builds more trust with users than pretending the model is more accurate than it is.

**Why SHAP over feature importance?**
Global feature importance tells you which features matter across all predictions. SHAP tells you which features drove *this specific* prediction — much more useful for a user who wants to understand why their particular laptop got that price.

---

## 📁 Dataset

> **This dataset is primary data — collected manually, not sourced from Kaggle or any third-party platform.**
>
> Laptop specs and prices were gathered directly from the Indian market, making this dataset original work. If you use it, please give credit.

- **Size:** 937 laptops (935 after deduplication)
- **Price range:** ₹22,990 – ₹7,79,900
- **Features:** 18 raw columns
- **Market:** Indian retail pricing

---

## 👨‍💻 Author

**Yash**

⭐ If you found this useful, star the repo on GitHub!

---

*Built as an end-to-end ML project — EDA → feature engineering → pipeline → deployment → explainability.*
