"""
Single authoritative run reproducing the exact sequence of the notebook,
so that all saved artifacts (clean data, figs, model, metrics) are
100% consistent with what the notebook shows.
Run with cwd = /home/claude/proj
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

RANDOM_STATE = 42

# ---------------- Checkpoint 2: clean ----------------
df_raw = pd.read_csv('house_sale.csv')
df = df_raw[df_raw['Kateqoriya'].isin(['Yeni tikili', 'Köhnə tikili'])].copy()

df['area_m2'] = df['Sahə'].str.extract(r'([\d.]+)').astype(float)
floors = df['Mərtəbə'].str.extract(r'(\d+)\s*/\s*(\d+)')
df['floor'] = pd.to_numeric(floors[0])
df['total_floors'] = pd.to_numeric(floors[1])
df['rooms'] = df['Otaq sayı']

df['has_repair'] = (df['Təmir'] == 'var').astype(int)
df['has_bill_of_sale'] = (df['Çıxarış'] == 'var').astype(int)
df['has_mortgage_option'] = (df['İpoteka'] == 'var').fillna(False).astype(int)
df['is_agency'] = (df['owner_title'] == 'vasitəçi (agent)').astype(int)
df['is_new_building'] = (df['Kateqoriya'] == 'Yeni tikili').astype(int)
df['is_vip'] = df['vip'].notna().astype(int)
df['is_featured'] = df['featured'].notna().astype(int)

critical = ['price', 'area_m2', 'floor', 'total_floors', 'rooms', 'lat', 'lng', 'location']
df = df.dropna(subset=critical)

df = df[df['floor'] <= df['total_floors']]
df = df[(df['area_m2'] >= 15) & (df['area_m2'] <= 500)]
df = df[(df['rooms'] >= 1) & (df['rooms'] <= 8)]

df['price_per_m2'] = df['price'] / df['area_m2']
lo, hi = df['price_per_m2'].quantile([0.01, 0.99])
df = df[(df['price_per_m2'] >= lo) & (df['price_per_m2'] <= hi)]
hi_abs = df['price'].quantile(0.995)
df = df[df['price'] <= hi_abs]

top_locations = df['location'].value_counts()
keep_locs = top_locations[top_locations >= 150].index
df['district'] = np.where(df['location'].isin(keep_locs), df['location'], 'Digər')

features = [
    'area_m2', 'rooms', 'floor', 'total_floors',
    'lat', 'lng', 'district', 'is_new_building',
    'has_repair', 'has_bill_of_sale', 'has_mortgage_option',
    'is_agency', 'is_vip', 'is_featured', 'views'
]
df_final = df[features + ['price']].copy()
df_final.to_csv('data/clean_apartments.csv', index=False)
print('CHECKPOINT2 clean shape:', df_final.shape, 'unique districts kept:', df['district'].nunique())

# ---------------- EDA figures ----------------
plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 10

na = df_raw.isna().sum()
na = na[na > 0].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(na.index[::-1], (na.values[::-1] / len(df_raw) * 100), color='#4C72B0')
ax.set_xlabel('Boş dəyərlərin faizi (%)'); ax.set_title('Xam datada boş dəyərlər')
plt.tight_layout(); plt.savefig('figs/01_missing_raw.png'); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(df_raw['price'].clip(upper=df_raw['price'].quantile(0.99)), bins=60, color='#dd8452')
axes[0].set_title('Xam data: price (99-cu persentilə qədər)'); axes[0].set_xlabel('Qiymət (AZN)')
axes[1].hist(df_final['price'], bins=60, color='#55a868')
axes[1].set_title('Təmizlənmiş data: price'); axes[1].set_xlabel('Qiymət (AZN)')
plt.tight_layout(); plt.savefig('figs/02_price_dist_before_after.png'); plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df_final['area_m2'], df_final['price'], s=4, alpha=0.25, color='#4C72B0')
ax.set_xlabel('Sahə (m²)'); ax.set_ylabel('Qiymət (AZN)'); ax.set_title('Qiymət vs Sahə')
plt.tight_layout(); plt.savefig('figs/03_price_vs_area.png'); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
order = sorted(df_final['rooms'].unique())
data = [df_final.loc[df_final['rooms'] == r, 'price'] for r in order]
ax.boxplot(data, tick_labels=[int(r) for r in order], showfliers=False)
ax.set_xlabel('Otaq sayı'); ax.set_ylabel('Qiymət (AZN)'); ax.set_title('Otaq sayına görə qiymət')
plt.tight_layout(); plt.savefig('figs/04_price_by_rooms.png'); plt.close()

num_cols_corr = ['area_m2', 'rooms', 'floor', 'total_floors', 'lat', 'lng', 'views',
                  'is_new_building', 'has_repair', 'has_bill_of_sale', 'has_mortgage_option',
                  'is_agency', 'is_vip', 'is_featured', 'price']
corr = df_final[num_cols_corr].corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols_corr))); ax.set_xticklabels(num_cols_corr, rotation=90)
ax.set_yticks(range(len(num_cols_corr))); ax.set_yticklabels(num_cols_corr)
for i in range(len(num_cols_corr)):
    for j in range(len(num_cols_corr)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=6)
fig.colorbar(im); ax.set_title('Korrelyasiya matrisi')
plt.tight_layout(); plt.savefig('figs/05_corr_heatmap.png'); plt.close()

top_districts = df_final['district'].value_counts().head(15).index
d = df_final[df_final['district'].isin(top_districts)].copy()
d['ppm2'] = d['price'] / d['area_m2']
avg = d.groupby('district')['ppm2'].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(avg.index, avg.values, color='#8172B2')
ax.set_xlabel('Orta qiymət (AZN/m²)'); ax.set_title('Rayonlara görə orta m² qiyməti (top-15)')
plt.tight_layout(); plt.savefig('figs/06_price_by_district.png'); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
data2 = [df_final.loc[df_final['is_new_building'] == 0, 'price'], df_final.loc[df_final['is_new_building'] == 1, 'price']]
ax.boxplot(data2, tick_labels=['Köhnə tikili', 'Yeni tikili'], showfliers=False)
ax.set_ylabel('Qiymət (AZN)'); ax.set_title('Bina növünə görə qiymət')
plt.tight_layout(); plt.savefig('figs/07_price_by_category.png'); plt.close()
print('EDA figures done')

# ---------------- Checkpoint 3: model comparison ----------------
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

X = df_final.drop(columns=['price'])
y = df_final['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
print('Train/Test:', X_train.shape, X_test.shape)

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
    'Random Forest (light)': RandomForestRegressor(n_estimators=80, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {'RMSE': 'neg_root_mean_squared_error', 'MAE': 'neg_mean_absolute_error', 'R2': 'r2'}
results = []
for name, model in models.items():
    pipe = Pipeline([('pre', make_pre()), ('model', model)])
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    results.append({'Model': name, 'CV RMSE': -cv_res['test_RMSE'].mean(),
                     'CV MAE': -cv_res['test_MAE'].mean(), 'CV R2': cv_res['test_R2'].mean()})
res_df = pd.DataFrame(results).sort_values('CV RMSE')
res_df.to_csv('data/model_comparison_results.csv', index=False)
print(res_df.to_string(index=False))

# ---------------- Checkpoint 4: tuning ----------------
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

pipe = Pipeline([('pre', make_pre()), ('model', HistGradientBoostingRegressor(random_state=RANDOM_STATE))])
param_dist = {
    'model__learning_rate': uniform(0.03, 0.22),
    'model__max_iter': randint(80, 200),
    'model__max_leaf_nodes': randint(15, 80),
    'model__min_samples_leaf': randint(10, 100),
    'model__l2_regularization': uniform(0.0, 1.0),
    'model__max_depth': randint(3, 12),
}
cv3 = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=8,
                             scoring='neg_root_mean_squared_error', cv=cv3,
                             random_state=RANDOM_STATE, n_jobs=1, verbose=0, refit=True)
search.fit(X_train, y_train)
print('Best CV RMSE:', -search.best_score_)
print('Best params:', search.best_params_)
best_model = search.best_estimator_

# ---------------- Checkpoint 5: final test eval ----------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import json

y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
metrics = {'test_RMSE': float(rmse), 'test_MAE': float(mae), 'test_R2': float(r2), 'test_MAPE_%': float(mape*100)}
print(json.dumps(metrics, indent=2))
with open('data/final_test_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

pred_df = pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred})
pred_df.to_csv('data/test_predictions.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(pred_df['y_true'], pred_df['y_pred'], s=6, alpha=0.25, color='#4C72B0')
lims = [0, max(pred_df['y_true'].max(), pred_df['y_pred'].max())]
axes[0].plot(lims, lims, 'r--', lw=1)
axes[0].set_xlabel('Həqiqi qiymət (AZN)'); axes[0].set_ylabel('Proqnoz qiymət (AZN)')
axes[0].set_title('Test set: Həqiqi vs Proqnoz')
resid = pred_df['y_true'] - pred_df['y_pred']
axes[1].hist(resid, bins=60, color='#55a868')
axes[1].axvline(0, color='r', linestyle='--', lw=1)
axes[1].set_xlabel('Xəta (Həqiqi - Proqnoz), AZN'); axes[1].set_title('Qalıqların paylanması')
plt.tight_layout(); plt.savefig('figs/08_test_actual_vs_pred.png'); plt.close()

from sklearn.inspection import permutation_importance
r_imp = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1, scoring='neg_root_mean_squared_error')
imp = pd.Series(r_imp.importances_mean, index=X_test.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp.index, imp.values, color='#c44e52')
ax.set_xlabel('RMSE artımı (permutation importance)'); ax.set_title('Dəyişənlərin əhəmiyyəti')
plt.tight_layout(); plt.savefig('figs/09_feature_importance.png'); plt.close()
print(imp.sort_values(ascending=False))

# Sample predictions for illustration
sample_pred = best_model.predict(X_test.iloc[:3])
print('Nümunə proqnozlar:', np.round(sample_pred, 0))
print('Həqiqi qiymətlər  :', y_test.iloc[:3].values)

# ---------------- Checkpoint 6: save model ----------------
joblib.dump(best_model, 'models/final_model_pipeline.joblib')
reloaded = joblib.load('models/final_model_pipeline.joblib')
print('Reload check:', reloaded.predict(X_test.iloc[:3]))

print('\n=== ALL DONE ===')
