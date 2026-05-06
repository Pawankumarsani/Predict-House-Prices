# Predict House Prices

A machine learning project that predicts California house prices using Linear Regression.

---

## About

This project builds a Linear Regression model trained on the California Housing Dataset. The goal is to predict house prices based on features like median income, house age, number of rooms, and location. It covers the full pipeline from loading data to evaluating the model and making new predictions.

---

## Dataset

The project uses the California Housing Dataset from scikit-learn. It contains 20,640 samples with 8 features.

| Feature | Description |
|---------|-------------|
| MedInc | Median income of the block |
| HouseAge | Median age of houses |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Total block population |
| AveOccup | Average household occupancy |
| Latitude | Block latitude |
| Longitude | Block longitude |
| Price | Median house value in $100,000s (target) |

---

## Requirements

```
numpy
scikit-learn
matplotlib
```

Install using:

```bash
pip install numpy scikit-learn matplotlib
```

---

## How It Works

The code is divided into the following steps:

**1. Load Data**
Loads the California Housing Dataset and separates features (X) and target prices (y).

**2. Train the Model**
Trains a Linear Regression model on the full dataset first to get baseline metrics, then again after a proper train/test split.

**3. Train/Test Split**
Splits the data into 80% training and 20% testing to evaluate how well the model generalizes to unseen data.

**4. Evaluate the Model**
Calculates R^2 Score, MSE, and MAE on both the training and test sets.

**5. Visualize**
Plots the actual data points (Median Income vs House Price) along with the regression line. New house predictions are marked directly on the plot.

**6. Predict New Houses**
Uses the trained model to predict prices for four new houses with different income levels.

---

## Results

### Full Data

| Metric | Value |
|--------|-------|
| R^2 Score | 0.6062 |
| MSE | 0.5243 |
| MAE | 0.5312 |

### Training Set

| Metric | Value |
|--------|-------|
| R^2 Score | 0.6126 |
| MSE | 0.5179 |
| MAE | 0.5286 |

### Test Set

| Metric | Value |
|--------|-------|
| R^2 Score | 0.5758 |
| MSE | 0.5559 |
| MAE | 0.5332 |

Training and test scores are close to each other which means the model is not overfitting.

---

## New House Predictions

| House Type | Income Level | Predicted Price |
|------------|-------------|-----------------|
| Budget | 2.5 | $149,570 |
| Mid-Range | 5.0 | $254,055 |
| Luxury | 8.0 | $379,436 |
| Premium | 10.0 | $463,024 |

---

## Plot

The visualization shows:

- Blue dots — actual data points from the dataset
- Red line — the regression line fitted by the model
- Colored markers — predicted prices for new houses
- Equation box — the regression equation printed on the chart

Regression equation:

```
Price = -0.570 + 0.451 x Median Income
```

---

## How to Run

**On Google Colab:**

Open the notebook and run all cells. Mount Google Drive when prompted.

**On a local machine:**

```bash
pip install numpy scikit-learn matplotlib jupyter
jupyter notebook Predict_House_Prices_.ipynb
```

Then run all cells from top to bottom.

---

## Notes

- Only Median Income is used for the 2D plot. All 8 features are used for training.
- The model explains around 60% of the variation in house prices which is reasonable for a simple linear model.
- For better accuracy, try Ridge Regression, Random Forest, or Gradient Boosting.
