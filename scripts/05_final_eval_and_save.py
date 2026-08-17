import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

X_test = pd.read_csv('/home/claude/proj/data/X_test.csv')
y_test = pd.read_csv('/home/claude/proj/data/y_test.csv').squeeze()

best_pipeline = joblib.load('/home/claude/proj/models/best_pipeline_tuned.joblib')

# ---- Final, one-time evaluation on the untouched hold-out test set ----
y_pred = best_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

metrics = {
    'test_RMSE': float(rmse),
    'test_MAE': float(mae),
    'test_R2': float(r2),
    'test_MAPE_%': float(mape * 100),
    'n_test': int(len(y_test)),
}
print(json.dumps(metrics, indent=2))

with open('/home/claude/proj/data/final_test_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

pred_df = pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred})
pred_df.to_csv('/home/claude/proj/data/test_predictions.csv', index=False)

# ---- Checkpoint 6: persist final model artifact ----
FINAL_MODEL_PATH = '/home/claude/proj/models/final_model_pipeline.joblib'
joblib.dump(best_pipeline, FINAL_MODEL_PATH)
print('Saved final model to', FINAL_MODEL_PATH)

# sanity-check reload
reloaded = joblib.load(FINAL_MODEL_PATH)
sample_pred = reloaded.predict(X_test.iloc[:3])
print('Reload sanity check predictions:', sample_pred)
