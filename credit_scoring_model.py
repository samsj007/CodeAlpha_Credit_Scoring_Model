import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def generate_synthetic_data(n_samples=1000):
    """Generates synthetic financial data for the model."""
    np.random.seed(42)
    data = {
        'income': np.random.randint(25000, 150000, n_samples),
        'debts': np.random.randint(0, 80000, n_samples),
        'late_payments': np.random.poisson(1, n_samples), # Payment history
    }
    df = pd.DataFrame(data)

    # Creating the Target Variable (1 = Default/High Risk, 0 = Good Credit)
    risk_factor = (df['debts'] / df['income']) + (df['late_payments'] * 0.4)
    df['credit_risk'] = (risk_factor > np.median(risk_factor) + 0.1).astype(int)
    
    return df

print("--- Starting Credit Scoring Model Pipeline ---")

# 1. Load Data
df = generate_synthetic_data(2000)

# 2. Feature Engineering
# Creating a new feature: Debt-to-Income Ratio
df['debt_to_income_ratio'] = df['debts'] / df['income']

X = df[['income', 'debts', 'late_payments', 'debt_to_income_ratio']]
y = df['credit_risk']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Model Training
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 6. Evaluation (Metrics required by CodeAlpha)
print("\n--- Model Evaluation Metrics ---")
print("Classification Report (Includes Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred, target_names=["Good Credit (0)", "High Risk (1)"]))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")