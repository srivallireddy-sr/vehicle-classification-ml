"""
Machine Learning Based Vehicle Classification for Indian Roads
Compares Logistic Regression, SVM and Random Forest on a binary
classification dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Data: 1000 samples, 20 features, 80/20 train-test split
# ---------------------------------------------------------------------------
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 1. Logistic Regression
# ---------------------------------------------------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

accuracy_log = accuracy_score(y_test, y_pred_log)
error_log = 1 - accuracy_log

print("Logistic Regression Accuracy:", accuracy_log)
print("Logistic Regression Error Rate:", error_log)
print(classification_report(y_test, y_pred_log))

fpr, tpr, _ = roc_curve(y_test, y_prob_log)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color="darkorange", linewidth=3,
         label="ROC Curve (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.text(0.6, 0.2, f"Accuracy = {accuracy_log:.2f}",
         fontsize=12, color="green", weight="bold")
plt.text(0.6, 0.1, f"Error Rate = {error_log:.2f}",
         fontsize=12, color="red", weight="bold")
plt.title("Logistic Regression Performance Analysis", fontsize=14, weight="bold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ---------------------------------------------------------------------------
# 2. Support Vector Machine (RBF kernel)
# ---------------------------------------------------------------------------
svm_model = SVC(kernel="rbf")
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
error_svm = 1 - accuracy_svm

print("SVM Accuracy:", accuracy_svm)
print("SVM Error Rate:", error_svm)
print(classification_report(y_test, y_pred_svm))

cm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
            linewidths=1, linecolor="black")
plt.title("SVM Confusion Matrix Heatmap", fontsize=14, weight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# ---------------------------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
error_rf = 1 - accuracy_rf
precision_rf = precision_score(y_test, y_pred_rf, average="weighted")
recall_rf = recall_score(y_test, y_pred_rf, average="weighted")
f1_rf = f1_score(y_test, y_pred_rf, average="weighted")

print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Error Rate:", error_rf)
print(classification_report(y_test, y_pred_rf))

metrics = [accuracy_rf, error_rf, precision_rf, recall_rf, f1_rf]
labels = ["Accuracy", "Error", "Precision", "Recall", "F1-Score"]
colors = ["#BEBBBB", "#737A5D", "#8E7766", "#899ABE", "#6B4D57"]

plt.figure(figsize=(10, 6))
plt.bar(labels, metrics, color=colors)
for i, value in enumerate(metrics):
    plt.text(i, value + 0.02, f"{value:.2f}", ha="center", fontsize=11, weight="bold")
plt.ylim(0, 1)
plt.title("Random Forest Performance Metrics", fontsize=15, weight="bold")
plt.ylabel("Score")
plt.grid(axis="y", alpha=0.3)
plt.show()

# ---------------------------------------------------------------------------
# 4. Performance comparison
# ---------------------------------------------------------------------------
models = ["Logistic Regression", "SVM", "Random Forest"]
accuracy = [accuracy_log, accuracy_svm, accuracy_rf]
error = [error_log, error_svm, error_rf]
x = np.arange(len(models))

plt.figure(figsize=(10, 6))
bars = plt.bar(x, accuracy, width=0.5)

# Highlight the best model
best_idx = int(np.argmax(accuracy))
bars[best_idx].set_edgecolor("black")
bars[best_idx].set_linewidth(2)

for i, value in enumerate(accuracy):
    plt.text(i, value + 0.02, f"{value:.2f}", ha="center", fontsize=11, weight="bold")
plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Model Performance Comparison (Accuracy)", fontsize=15, weight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(models, accuracy, marker="o", linewidth=3, label="Accuracy")
plt.plot(models, error, marker="s", linestyle="--", label="Error Rate")
for i, value in enumerate(accuracy):
    plt.text(i, value + 0.02, f"{value:.2f}", ha="center")
for i, value in enumerate(error):
    plt.text(i, value - 0.05, f"{value:.2f}", ha="center")
plt.title("Accuracy vs Error Rate Comparison", fontsize=14, weight="bold")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"\nBest model: {models[best_idx]} (accuracy = {accuracy[best_idx]:.3f})")
