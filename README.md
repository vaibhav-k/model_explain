# Model Explain

---

[![Downloads](https://pepy.tech/badge/model-explain)](https://pepy.tech/project/model-explain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/model-explain.svg)](https://badge.fury.io/py/model-explain)

---

`model_explain` is a Python package designed for providing **interpretability** and **explainability** for machine learning models. It supports a variety of popular techniques for explaining individual model predictions, making it easier to understand how black-box models (such as neural networks, decision trees, and ensemble models) make decisions.

The library is model-agnostic, meaning it can be used with any machine learning model, regardless of the underlying algorithm. With built-in support for **LIME** and **SHAP**, it offers powerful tools to interpret and explain the contributions of individual features in model predictions.

## Features

- **Model-Agnostic Explanation Techniques**: 
  - Supports popular model-agnostic explanation techniques like **LIME** (Local Interpretable Model-agnostic Explanations) and **SHAP** (SHapley Additive exPlanations).
  
- **Compatibility with Multiple ML Frameworks**:
  - Works with a wide range of machine learning models from libraries such as `scikit-learn`, `XGBoost`, `LightGBM`, and more.
  
- **Visualizations**:
  - Provides easy-to-use visualizations to display feature importance, which helps in understanding how model predictions are affected by various input features.
  
- **Interpretability for Individual Predictions**:
  - Offers tools to explain specific predictions, making it suitable for tasks that require model transparency (e.g., finance, healthcare).

- **Simple Integration**:
  - Designed to integrate seamlessly with your existing machine learning pipelines.

## Installation

You can install `model_explain` via pip directly from GitHub:

```bash
pip install model-explain
```

Or, if you prefer to clone the repository and install manually:

```bash
git clone https://github.com/vaibhav-k/model_explain.git
cd model_explain
pip install .
```

## Usage
### 1. LIME (Local Interpretable Model-agnostic Explanations)

LIME explains individual predictions by approximating the model locally using interpretable models like linear regression or decision trees. It creates a local surrogate model around the prediction and uses it to explain how the model reached its decision.

```python
from model_explain import lime

explainer = lime.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=class_names,
    mode="classification",
)

explanation = explainer.explain_instance(X_test.iloc[0], model.predict_proba)
explanation.show_in_notebook()
```

### 2. SHAP (SHapley Additive exPlanations)

SHAP uses game theory to calculate the contribution of each feature to a given prediction. It provides both local and global interpretability by calculating feature importance for individual predictions or across an entire dataset.

```python
import shap

# Initialize SHAP explainer
explainer = shap.Explainer(model)

# Get SHAP values for a set of predictions
shap_values = explainer(X_test)

# Visualize the SHAP values
shap.summary_plot(shap_values, X_test)
```

### 3. Model Interpretation

You can use model_explain to interpret the feature importance of any machine learning model.

```python
from model_explain import feature_importance

# Get feature importance from a trained model
importance = feature_importance(model, X_train)

# Visualize feature importance
importance.plot()
```

## Supported Models
- **scikit-learn** models (e.g., LogisticRegression, RandomForestClassifier, SVM)
- **XGBoost** models (xgb.XGBClassifier, xgb.XGBRegressor)
- **LightGBM** models (lgb.LGBMClassifier, lgb.LGBMRegressor)
- Other models compatible with scikit-learn interfaces.

## Key Concepts

### LIME

LIME is an explanation technique that focuses on explaining individual predictions rather than global model behavior. It works by perturbing the input data and training a simple, interpretable model on the perturbed data to approximate the decision boundary of the complex model locally.

### SHAP

SHAP values are based on cooperative game theory and attribute the prediction to features by computing the average contribution of each feature to the prediction across all possible feature subsets.

### Visualizations

Both LIME and SHAP provide built-in visualizations for explaining models:

SHAP Summary Plot: A global view of feature importance across the dataset.

LIME Explanation Plot: A local view of how features influenced a specific prediction.

Example usage for visualizing SHAP values:

```python
import shap
shap.initjs()
shap.summary_plot(shap_values, X_test)
```

## Use Cases

- **Model Debugging**: Identify features or model behaviors that might be causing bias or overfitting.
- **Feature Importance**: Determine which features are contributing most to model predictions.
- **Improving Trust in Models**: Make machine learning models more transparent, especially in regulated industries like finance or healthcare.
- *Understanding Black-box Models*: Get insights into deep learning models, ensemble methods, and other complex algorithms.

## Contributing

Contributions to `model_explain` are welcome! Feel free to fork the repository and submit a pull request.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add feature'`)
5. Push to the branch (`git push origin feature-name`)
6. Open a pull request

Please read `CONTRIBUTING.md` for more details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
