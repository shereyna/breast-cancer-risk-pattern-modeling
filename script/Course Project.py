# Course Project - Shereyna Shinbo

# Import CSV file
import pandas as pd
data = pd.read_csv('/Users/shereynashinbo/Documents/GRAD 50800/Course Project/dataset.csv')

# Inspect the data
len(data)

# Replace missing values
import numpy as np
data['bmi_c'] = data['bmi_c'].replace(-99, np.nan)
for col in ['compfilm_c','famhx_c','hrt_c','prvmam_c','biophx_c']:
    data[col] = data[col].replace(9, np.nan)

# Drop columns
data = data.drop(['CaTypeO', 'ptid'], axis=1)

# Drop missing data
data = data.dropna()

len(data)

# View data
data.head()
data.info()

# Change data type
col = ['compfilm_c', 'famhx_c', 'hrt_c', 'prvmam_c', 'biophx_c']
data[col] = data[col].astype('int64')

pos = (data['cancer_c'] == 1).sum()

# target and predictor
X = data.drop(columns=['cancer_c'])
y = data['cancer_c']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# evaluation
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, proba)

acc, auc  # (0.9947229551451188, 0.9108711870026526)

# coefficients
coeffs = pd.DataFrame({
    'variable': X.columns,
    'coefficient': model.coef_[0]
})
coeffs

joblib.dump(model, 'logistic_model.joblib')

coeffs['odds_ratio'] = np.exp(coeffs['coefficient'])
coeffs

coeffs_sorted = coeffs.sort_values(by='coefficient', ascending=False)
coeffs_sorted

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.barh(coeffs_sorted["variable"], coeffs_sorted["odds_ratio"])
plt.axvline(1.0, linestyle="--")
plt.xlabel("Odds ratio")
plt.title("Logistic Regression: Odds Ratios by Predictor")
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Logistic Regression: Confusion Matrix")
plt.tight_layout()
plt.show()


# Random Forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

X = data.drop(columns=['cancer_c'])
y = data['cancer_c']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:,1]

rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

# feature importance
rf_importance = pd.DataFrame({
    'variable': X.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

rf_acc, rf_auc
rf_importance

# Visualize
import matplotlib.pyplot as plt

# Assumes rf_importance has columns ["variable", "importance"]
rf_sorted = rf_importance.sort_values(by="importance", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(rf_sorted["variable"], rf_sorted["importance"])
plt.xlabel("Importance")
plt.title("Random Forest: Feature Importances")
plt.tight_layout()
plt.show()

