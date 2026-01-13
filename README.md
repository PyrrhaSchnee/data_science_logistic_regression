# 🧙‍♂️ Hogwarts House Sorting Hat - Logistic Regression from Scratch

A complete implementation of a **multi-class logistic regression classifier** built entirely from scratch in Python. This project demonstrates the ability to classify students into Hogwarts Houses (Gryffindor, Hufflepuff, Ravenclaw, Slytherin) based on their academic performance across 13 different courses.

> **Note:** This project was developed as a data science exercise to deeply understand the mathematical foundations of machine learning algorithms without relying on high-level libraries like scikit-learn. 

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Implementation](#technical-implementation)
- [Visualizations](#visualizations)
- [Skills Demonstrated](#skills-demonstrated)
- [License](#license)

---

## 🎯 Overview

This project implements a **one-vs-all (OvA) logistic regression** classifier to solve a multi-class classification problem.  The goal is to predict which Hogwarts House a student belongs to based on their grades in various magical subjects.

### Key Highlights
- **No ML libraries used** - All algorithms implemented from scratch using only NumPy for numerical operations
- **Complete ML pipeline** - Data preprocessing, feature engineering, model training, and prediction
- **Exploratory Data Analysis** - Histogram, scatter plot, and pair plot visualizations
- **Production-ready code** - Clean, documented, and type-annotated Python code

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Custom `describe` function** | Replicates pandas' `describe()` from scratch with Count, Mean, Std, Min, 25%, 50%, 75%, Max |
| **Data Visualization** | Histograms, scatter plots, and pair plots for feature analysis |
| **One-vs-All Classification** | Multi-class classification using binary logistic regression |
| **Gradient Descent** | Batch, stochastic, and mini-batch gradient descent implementations |
| **Feature Standardization** | Z-score normalization for stable training |
| **Missing Value Imputation** | Mean imputation for handling incomplete data |
| **Momentum Optimization** | Optional momentum-based gradient descent for faster convergence |

---

## 📁 Project Structure

```
data_science_logistic_regression/
├── describe.py          # Custom statistical description (like pandas describe)
├── histogram.py         # Histogram visualization for feature analysis
├── scatter_plot.py      # Scatter plot for feature correlation analysis
├── pair_plot.py         # Pair plot for multi-feature visualization
├── logreg_train.py      # Training script for logistic regression model
├── logreg_predict.py    # Prediction script using trained model
├── utilities.py         # Shared utility functions and data structures
├── dataset_train.csv    # Training dataset
├── dataset_test.csv     # Test dataset for predictions
├── Makefile             # Build automation
├── requirements. txt     # Python dependencies
├── . flake8              # Linting configuration
├── pyproject.toml       # Project configuration
└── LICENSE              # MIT License
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/PyrrhaSchnee/data_science_logistic_regression.git
cd data_science_logistic_regression

# Create virtual environment (optional but recommended)
python3 -m venv . venv
source . venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Or use the provided setup script: 
```bash
make prep
```

---

## 💻 Usage

### 1. Data Description
Generate statistical summary of the dataset: 
```bash
python3 describe.py dataset_train.csv
# or
make describe
```

### 2. Exploratory Data Analysis

```bash
# Generate histogram
make histogram

# Generate scatter plot
make scatter

# Generate pair plot
make pairplot
```

### 3. Train the Model
```bash
python3 logreg_train.py dataset_train.csv
# or
make train
```

This generates `after_train.csv` containing: 
- Feature names
- Feature means and standard deviations (for standardization)
- Learned theta parameters for each house

### 4. Make Predictions
```bash
python3 logreg_predict.py dataset_test.csv
# or
make predict
```

Outputs `houses.csv` with predicted houses for each student.

---

## 🔬 Technical Implementation

### Algorithm Overview

The classifier uses **One-vs-All (OvA)** strategy with logistic regression:

1. **Data Preprocessing**
   - Parse CSV and extract numeric features
   - Handle missing values with mean imputation
   - Standardize features using Z-score normalization:  `x' = (x - μ) / σ`
   - Add bias term (intercept)

2. **Training (One-vs-All)**
   - Train 4 binary classifiers (one per house)
   - Use sigmoid function:  `σ(z) = 1 / (1 + e^(-z))`
   - Optimize using gradient descent with configurable: 
     - Learning rate (default: 0.1)
     - Epochs (default: 1500)
     - Batch size (batch/SGD/mini-batch)
     - Momentum (optional)

3. **Prediction**
   - Compute scores for all 4 classifiers
   - Assign to house with highest score

### Mathematical Foundations

**Logistic Regression Hypothesis:**
```
h_θ(x) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
```

**Gradient Descent Update:**
```
θⱼ := θⱼ - α · (1/m) · Σ(h_θ(xⁱ) - yⁱ) · xⱼⁱ
```

**Numerically Stable Sigmoid:**
```python
def sigmoid(z:  float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    else: 
        return math.exp(z) / (1.0 + math.exp(z))
```

---

## 📊 Visualizations

The project includes tools for exploratory data analysis: 

| Tool | Purpose |
|------|---------|
| `histogram.py` | Analyze grade distribution across houses to identify homogeneous features |
| `scatter_plot.py` | Identify correlated/similar features |
| `pair_plot.py` | Comprehensive pairwise feature analysis |

---

## 🛠 Skills Demonstrated

This project showcases proficiency in: 

### Data Science & Machine Learning
- ✅ Logistic Regression implementation from scratch
- ✅ Gradient Descent optimization (Batch, SGD, Mini-batch)
- ✅ Feature engineering and standardization
- ✅ Missing data handling (imputation)
- ✅ Multi-class classification (One-vs-All)
- ✅ Model evaluation and accuracy metrics

### Software Engineering
- ✅ Clean, modular Python code with type annotations
- ✅ Comprehensive docstrings and documentation
- ✅ Build automation with Makefile
- ✅ Code quality tools (flake8, black)
- ✅ Virtual environment management
- ✅ Version control with Git

### Data Visualization
- ✅ Matplotlib for statistical visualizations
- ✅ Exploratory Data Analysis (EDA)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

---

## 👤 Author

**PyrrhaSchnee**

- GitHub: [@PyrrhaSchnee](https://github.com/PyrrhaSchnee)

---

<p align="center">
  <i>Built with ❤️ and pure Python mathematics</i>
</p>
