import time
import joblib
import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_STATE = 42

X_train = pd.read_csv('/home/claude/proj/data/X_train.csv')
y_train = pd.read_csv('/home/claude/proj/data/y_train.csv').squeeze()

num_cols = [c for c in X_train.columns if c != 'district']
cat_cols = ['district']

pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

pipe = Pipeline([
    ('pre', pre),
    ('model', HistGradientBoostingRegressor(random_state=RANDOM_STATE))
])

param_dist = {
    'model__learning_rate': uniform(0.03, 0.22),
    'model__max_iter': randint(80, 200),
    'model__max_leaf_nodes': randint(15, 80),
    'model__min_samples_leaf': randint(10, 100),
    'model__l2_regularization': uniform(0.0, 1.0),
    'model__max_depth': randint(3, 12),
}

# NOTE: environment has a single CPU core, so n_jobs=-1 gives no real
# parallel speed-up here. We keep the search budget modest (n_iter x cv)
# specifically to avoid the multi-hour Colab-style runtime problem.
cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=8,
    scoring='neg_root_mean_squared_error',
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=1,
    verbose=2,
    refit=True,
)

t0 = time.time()
search.fit(X_train, y_train)
t1 = time.time()
print('RandomizedSearchCV total time (s):', round(t1 - t0, 1))
print('Best CV RMSE:', -search.best_score_)
print('Best params:', search.best_params_)

joblib.dump(search.best_estimator_, '/home/claude/proj/models/best_pipeline_tuned.joblib')

cvres = pd.DataFrame(search.cv_results_).sort_values('rank_test_score')
cvres.to_csv('/home/claude/proj/data/randomsearch_cv_results.csv', index=False)
print(cvres[['params', 'mean_test_score', 'rank_test_score']].head(10).to_string())
