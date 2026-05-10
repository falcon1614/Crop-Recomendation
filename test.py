# =============================================================================
# 1. Import Libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# =============================================================================
# 2. Load Dataset
# =============================================================================
crop = pd.read_csv('Crop_recommendation.csv')
print("Dataset shape:", crop.shape)
crop.head()


# =============================================================================
# 3. Data Overview
# =============================================================================
crop.info()
print("\nMissing values:\n", crop.isnull().sum())
print("\nDuplicate rows:", crop.duplicated().sum())


# =============================================================================
# 4. Statistical Summary
# =============================================================================
crop.describe()


# =============================================================================
# 5. Class Distribution
# =============================================================================
crop['label'].value_counts()


# =============================================================================
# 6. Encode Labels
# =============================================================================
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14,
    'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)


# =============================================================================
# 7. Drop Original Label Column
# =============================================================================
crop.drop('label', axis=1, inplace=True)
crop.head()


# =============================================================================
# 8. Feature / Target Split
# =============================================================================
X = crop.drop('crop_num', axis=1)
y = crop['crop_num']


# =============================================================================
# 9. Train/Test Split (Stratified)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =============================================================================
# 10. Feature Scaling
# =============================================================================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)      # <-- correct: transform, not fit_transform

# =============================================================================
# 11. Cross‑Validation on Base Random Forest
# =============================================================================
base_rf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(base_rf, X_train_scaled, y_train, cv=5)
print("Base RF CV scores:", cv_scores)
print("Mean CV accuracy: {:.4f}".format(cv_scores.mean()))


# =============================================================================
# 12. Train and Evaluate Multiple Classifiers (Before Tuning)
# =============================================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Test Accuracy: {acc:.4f}")


# =============================================================================
# 13. HYPERPARAMETER TUNING FOR RANDOM FOREST (GridSearchCV)
# =============================================================================
print("\n=== Starting Hyperparameter Tuning ===")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_scaled, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross‑Validation Accuracy: {:.4f}".format(grid_search.best_score_))


# =============================================================================
# 14. Evaluate Tuned Random Forest on Test Set
# =============================================================================
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)
tuned_acc = accuracy_score(y_test, y_pred_rf)
print("Tuned Random Forest Test Accuracy: {:.4f}".format(tuned_acc))


# =============================================================================
# 15. Confusion Matrix – Tuned Random Forest
# =============================================================================
crop_dict_rev = {v: k for k, v in crop_dict.items()}
crop_names = [crop_dict_rev[i] for i in range(1, 23)]

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=crop_names, yticklabels=crop_names,
            linewidths=0.5, linecolor='white')
plt.title(f'Tuned Random Forest Confusion Matrix (Acc: {tuned_acc:.4f})', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=90, fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()


# =============================================================================
# 16. Model Accuracy Comparison (All Models Including Tuned RF)
# =============================================================================
# Re‑evaluate base models
model_names = list(models.keys())
base_accs = []
for name in model_names:
    model = models[name]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    base_accs.append(accuracy_score(y_test, y_pred))

# Add tuned RF
all_names = model_names + ['Tuned RF']
all_accs = base_accs + [tuned_acc]
acc_pct = [a*100 for a in all_accs]

colors = ['#A5D6A7']*len(model_names) + ['#2E7D32']

plt.figure(figsize=(12, 6))
bars = plt.bar(all_names, acc_pct, color=colors, edgecolor='white', linewidth=0.8)
for bar, acc in zip(bars, acc_pct):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
plt.title('Model Accuracy Comparison (Including Tuned Random Forest)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
# 17. Feature Correlation Heatmap
# =============================================================================
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
corr = crop[features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()


# =============================================================================
# 18. Feature Importance from Tuned Random Forest
# =============================================================================
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in indices]
sorted_importances = importances[indices]

plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importances)), sorted_importances,
        color=plt.cm.Greens(np.linspace(0.8, 0.3, len(sorted_importances))))
plt.xticks(range(len(sorted_importances)), sorted_features, rotation=45, ha='right')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance (Tuned Model)')
plt.tight_layout()
plt.show()


# =============================================================================
# 19. Save Model and Scaler for Deployment
# =============================================================================
pickle.dump(best_rf, open('model_tuned.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("Tuned model and scaler saved.")


# =============================================================================
# 20. Recommendation Function (CORRECTED)
# =============================================================================
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict the best crop for given soil and climate parameters.
    The scaler must be fitted on the training data beforehand.
    """
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)          
    pred_num = best_rf.predict(input_scaled)[0]
    return crop_dict_rev[pred_num]

# =============================================================================
# 21. Test the Recommendation System
# =============================================================================
test_samples = [
    (40, 50, 50, 40.0, 20.0, 100.0, 100.0),
    (100, 90, 100, 50.0, 90.0, 100.0, 202.0),
    (10, 10, 10, 15.0, 80.0, 4.5, 10.0)
]

for i, params in enumerate(test_samples, 1):
    crop_rec = recommend_crop(*params)
    print(f"Sample {i} → Recommended crop: {crop_rec}")

for i, params in enumerate(test_samples, 1):
    crop_rec = recommend_crop(*params)
    print(f"Sample {i} → Recommended crop: {crop_rec}")
