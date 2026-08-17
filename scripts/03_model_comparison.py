import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

RANDOM_STATE = 42

df = pd.read_csv('/home/claude/proj/data/clean_apartments.csv')
X = df.drop(columns=['price'])
y = df['price']

# Single held-out test set - locked away until Checkpoint 5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE
)
X_train.to_csv('/home/claude/proj/data/X_train.csv', index=False)
X_test.to_csv('/home/claude/proj/data/X_test.csv', index=False)
y_train.to_csv('/home/claude/proj/data/y_train.csv', index=False)
y_test.to_csv('/home/claude/proj/data/y_test.csv', index=False)

num_cols = [c for c in X.columns if c != 'district']
cat_cols = ['district']

def make_pre():
    return ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

models = {
    'Ridge Regression': Ridge(random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=RANDOM_STATE),
    'Random Forest (light)': RandomForestRegressor(
        n_estimators=80, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
    ),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    'RMSE': 'neg_root_mean_squared_error',
    'MAE': 'neg_mean_absolute_error',
    'R2': 'r2',
}

results = []
timings = {}
for name, model in models.items():
    pipe = Pipeline([('pre', make_pre()), ('model', model)])
    t0 = time.time()
    cv_res = cross_validate(
        pipe, X_train, y_train, cv=cv, scoring=scoring,
        n_jobs=-1, return_train_score=False
    )
    t1 = time.time()
    timings[name] = round(t1 - t0, 1)
    results.append({
        'Model': name,
        'CV RMSE (mean)': -cv_res['test_RMSE'].mean(),
        'CV RMSE (std)': cv_res['test_RMSE'].std(),
        'CV MAE (mean)': -cv_res['test_MAE'].mean(),
        'CV R2 (mean)': cv_res['test_R2'].mean(),
        'CV R2 (std)': cv_res['test_R2'].std(),
        'CV time (s, 5 folds)': timings[name],
    })

res_df = pd.DataFrame(results).sort_values('CV RMSE (mean)')
res_df.to_csv('/home/claude/proj/data/model_comparison_results.csv', index=False)
pd.set_option('display.width', 140)
print(res_df.to_string(index=False))
print()
print('Total CV wall time (s):', sum(timings.values()))
