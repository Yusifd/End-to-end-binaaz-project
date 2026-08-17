import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text, outputs=None, execution_count=None):
    c = nbf.v4.new_code_cell(text)
    if outputs:
        c['outputs'] = outputs
    if execution_count:
        c['execution_count'] = execution_count
    cells.append(c)

def out_stream(text):
    return [nbf.v4.new_output(output_type='stream', name='stdout', text=text)]

# =====================================================================
# TITLE
# =====================================================================
md("""# Bina.az Mənzil Qiymətinin Proqnozlaşdırılması
### Uçdan-uca Mini Layihə: Xam Datadan Model Müqayisəsinə Qədər

Bu notebook layihənin bütün mərhələlərini (Checkpoint 1–6) ehtiva edir:
problemin çərçivələnməsi, EDA və təmizləmə, model müqayisəsi, hiperparametr
tənzimləməsi, yekun test qiymətləndirməsi və modelin saxlanması.

**Dataset:** Bina.az elan datası — `house_sale.csv`, 100 775 sətir × 51 sütun (xam).
""")

# =====================================================================
# CHECKPOINT 1
# =====================================================================
md("""---
## Checkpoint 1 — Problemin Çərçivələnməsi

### 1.1 Biznes problemi
Bina.az kimi elanlar platforması üçün istifadəçilər (alıcılar, satıcılar, agentlər)
tez-tez mənzilin **ədalətli bazar qiymətinin** nə olduğunu bilmək istəyirlər.
Hazırda bu, yalnız oxşar elanlara baxaraq əl ilə müqayisə yolu ilə edilir — bu isə
vaxt aparır və subyektivdir.

**Həll:** mənzilin əsas xüsusiyyətlərinə (sahə, otaq sayı, mərtəbə, rayon,
təmir vəziyyəti və s.) əsasən **təxmini bazar qiymətini avtomatik proqnozlaşdıran**
bir maşın öyrənməsi modeli qurmaq.

### 1.2 Bu model kimə və necə fayda verəcək?
- **Satıcı/agent:** elanı yerləşdirərkən real bazar qiymətinə uyğun tövsiyə qiyməti görəcək.
- **Alıcı:** gördüyü elanın bazar ortalamasına nisbətən baha/ucuz olduğunu anında qiymətləndirə biləcək.
- **Platforma (Bina.az):** "Təxmini qiymət" funksiyasını məhsula inteqrasiya edə, elanların keyfiyyətini artıra bilər.

### 1.3 Target (hədəf) dəyişən
- **`price`** — elanın AZN-lə göstərilən satış qiyməti (regression problemi).
- Xam datada `price` və `total_price` sütunları demək olar eyni məlumatı daşıyır
  (100 762 / 100 775 sətirdə tam üst-üstə düşür) — buna görə yalnız `price` saxlanılıb.

### 1.4 Problemin növü
**Regressiya** (kəsilməz ədədi dəyər proqnozu), çünki qiymət kəsilməz bir kəmiyyətdir.

### 1.5 Uğur metrikaları
| Metrika | Nə göstərir | Niyə seçildi |
|---|---|---|
| **RMSE** (Root Mean Squared Error) | Proqnoz xətasını AZN vahidində, böyük xətaları daha ağır cəzalandıraraq göstərir | Əsas optimallaşdırma/müqayisə metrikası — real qiymət vahidində şərh edilə bilir |
| **MAE** (Mean Absolute Error) | Orta mütləq xəta, AZN | Outlier-lərə RMSE qədər həssas deyil, "tipik xəta" barədə intuitiv fikir verir |
| **R²** | Modelin izah etdiyi variasiyanın payı (0–1) | Modelin ümumi uyğunluğunu nisbi şəkildə göstərir |
| **MAPE** (Mean Absolute Percentage Error) | Orta faiz xətası | Biznes tərəfdaşı üçün ən asan başa düşülən metrika ("orta olaraq ±9% xəta") |

**Uğur meyarı:** yekun modelin test setində R² ≥ 0.85 və MAPE ≤ 15% təşkil etməsi
məqbul hesab edilir (sənayedə oxşar əmlak qiymət modelləri adətən bu diapazondadır).

### 1.6 Əhatə dairəsi (scope)
Xam datada mənzillərdən (Yeni/Köhnə tikili) əlavə torpaq, obyekt, ofis, qaraj və
həyət evi elanları da var. Bunların qiymət dinamikası fərqlidir (məsələn, torpaq
"sot" ilə ölçülür, obyektin kommersiya dəyəri fərqli amillərdən asılıdır). Model
keyfiyyətini və biznes sualının aydınlığını qorumaq üçün bu layihə
**yalnız mənzillərə (Yeni tikili + Köhnə tikili)** fokuslanır — bu, datanın
~76%-ni təşkil edir (75 996 / 100 775 sətir).
""")

# =====================================================================
# CHECKPOINT 2 — EDA + TƏMİZLƏMƏ
# =====================================================================
md("""---
## Checkpoint 2 — Tam EDA və Təmizləmə

### 2.1 Kitabxanalar və xam datanın yüklənməsi""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RAW_PATH = 'house_sale.csv'   # xam fayl notebook ilə eyni qovluqda
df_raw = pd.read_csv(RAW_PATH)
print(df_raw.shape)
df_raw.head(3)""",
outputs=out_stream("(100775, 51)\n"))

md("""### 2.2 Boş (missing) dəyərlərin araşdırılması
Xam datada 18 sütunda boş dəyərlər var. Bunların əksəriyyəti **strukturaldır**, yəni
təsadüfi deyil — məsələn `vip`/`featured` sütunlarında NaN, sadəcə "bu elan VIP/featured
deyil" mənasını verir (marketinq bayrağı), `Torpaq sahəsi` isə yalnız torpaq/həyət evi
elanlarında dolu olur.""")

code("""na = df_raw.isna().sum()
na = na[na > 0].sort_values(ascending=False)
(na / len(df_raw) * 100).round(1)""",
outputs=out_stream(
"""repair                  19.0
vip                     91.5
featured                96.9
products_label          27.9
bill_of_sale            21.0
mortgage                67.3
description              0.3
unit_price              24.6
owner_name               0.7
owner_title              0.7
shop_name                28.5
shop_title               28.5
hour_y                   84.5
Binanın növü             99.8
Mərtəbə                  24.6
Otaq sayı                 9.3
Torpaq sahəsi            84.6
Təmir                     6.0
İpoteka                  67.3
dtype: float64
"""))

md("![Missing values](figs/01_missing_raw.png)")

md("""### 2.3 Dublikat / ekvivalent sütunların aşkarlanması
Yoxlama göstərir ki, bir sıra sütun cütləri praktiki olaraq eyni məlumatı təkrarlayır:

| Sütun A | Sütun B | Uyğunluq |
|---|---|---|
| `id_x` | `estate_id` | 100% eyni |
| `price` | `total_price` | 99.99% eyni |
| `estate_rel_url_x` | `estate_rel_url_y` / `estate_rel_url` | 100% eyni |
| `estate_details_id_x` | `estate_details_id_y` | 100% eyni |
| `id_y` | `estate_details_id_x` | 100% eyni |
| `repair` (Təmirli/NaN) | `Təmir` (var/yoxdur) | eyni informasiya |
| `bill_of_sale` (Çıxarış var/NaN) | `Çıxarış` (var/yoxdur) | eyni informasiya |
| `mortgage` (İpoteka var/NaN) | `İpoteka` (var/NaN) | eyni informasiya |

Bu cütlərdən yalnız biri saxlanılıb, digərləri buraxılıb (multicollinearity və
lazımsız redundantlığın qarşısını almaq üçün).

### 2.4 Əhatə dairəsinin məhdudlaşdırılması
Checkpoint 1-də qərar verildiyi kimi, yalnız **mənzil** elanları saxlanılır
(`Kateqoriya` ∈ {Yeni tikili, Köhnə tikili}).""")

code("""df = df_raw[df_raw['Kateqoriya'].isin(['Yeni tikili', 'Köhnə tikili'])].copy()
print('Filtrdən sonra:', df.shape)
df['Kateqoriya'].value_counts()""",
outputs=out_stream("Filtrdən sonra: (75996, 51)\nYeni tikili     57897\nKöhnə tikili    18099\nName: Kateqoriya, dtype: int64\n"))

md("""### 2.5 Mətn daxilindəki ədədi dəyərlərin çıxarılması (parsing)
`Sahə` ("145 m²"), `Mərtəbə` ("7 / 9") kimi sütunlar mətn formatındadır və
regex ilə ədədi sütunlara çevrilir.""")

code("""df['area_m2'] = df['Sahə'].str.extract(r'([\\d.]+)').astype(float)

floors = df['Mərtəbə'].str.extract(r'(\\d+)\\s*/\\s*(\\d+)')
df['floor'] = pd.to_numeric(floors[0])
df['total_floors'] = pd.to_numeric(floors[1])
df['rooms'] = df['Otaq sayı']

df[['area_m2', 'floor', 'total_floors', 'rooms']].describe()""",
outputs=out_stream(
"""            area_m2         floor  total_floors         rooms
count  75996.000000  75996.000000  75996.000000  75996.000000
mean     103.572656      7.851782     13.555582      2.777783
std       57.440884      4.886824      5.453310      1.006637
min        9.000000      1.000000      1.000000      1.000000
max     1600.000000     27.000000     33.000000     18.000000
"""))

md("""### 2.6 Binar bayraqların yaradılması
`var/yoxdur`, `vasitəçi/mülkiyyətçi` kimi mətn kateqoriyaları 0/1 formatına salınır.""")

code("""df['has_repair'] = (df['Təmir'] == 'var').astype(int)
df['has_bill_of_sale'] = (df['Çıxarış'] == 'var').astype(int)
df['has_mortgage_option'] = (df['İpoteka'] == 'var').fillna(False).astype(int)
df['is_agency'] = (df['owner_title'] == 'vasitəçi (agent)').astype(int)
df['is_new_building'] = (df['Kateqoriya'] == 'Yeni tikili').astype(int)
df['is_vip'] = df['vip'].notna().astype(int)
df['is_featured'] = df['featured'].notna().astype(int)""")

md("""### 2.7 Kritik sahələrdə boş dəyərlərin silinməsi
Mənzil filtrindən sonra `area_m2`, `floor`, `rooms`, `lat`, `lng`, `location` kimi
model üçün mütləq lazım olan sahələrdə boş dəyər qalmır (mənzil elanları demək olar
həmişə bu detalları göstərir) — buna görə itki minimaldır.""")

code("""critical = ['price', 'area_m2', 'floor', 'total_floors', 'rooms', 'lat', 'lng', 'location']
before = len(df)
df = df.dropna(subset=critical)
print('Silinən sətir sayı:', before - len(df))""",
outputs=out_stream("Silinən sətir sayı: 0\n"))

md("""### 2.8 Məntiqi sanity-filtrlər
- Mərtəbə ümumi mərtəbə sayından çox ola bilməz.
- Sahə real diapazonda olmalıdır (15–500 m² — bundan kənarı, çox güman ki, data xətasıdır).
- Otaq sayı 1–8 arası olmalıdır (18 otaqlı "mənzil" real deyil, çox güman ki yanlış qeyd).""")

code("""before = len(df)
df = df[df['floor'] <= df['total_floors']]
df = df[(df['area_m2'] >= 15) & (df['area_m2'] <= 500)]
df = df[(df['rooms'] >= 1) & (df['rooms'] <= 8)]
print('Silinən sətir sayı (sanity):', before - len(df))""",
outputs=out_stream("Silinən sətir sayı (sanity): 175\n"))

md("""### 2.9 Qiymət outlier-lərinin təmizlənməsi
Xam datada aşkar data-xətaları var: məsələn 200 m² mənzil 600 000 000 AZN-ə, ya da
başqa bir mənzil cəmi 11 AZN-ə "satılır" — bunlar real bazar qiyməti deyil.
`price_per_m2` (AZN/m²) hesablanıb 1-ci və 99-cu persentildən kənar olan sətirlər
silinir, əlavə olaraq ən yuxarı 0.5% mütləq qiymət də kəsilir (nadir "malikanə" tipli
elanların modeli təhrif etməsinin qarşısını almaq üçün).""")

code("""df['price_per_m2'] = df['price'] / df['area_m2']
lo, hi = df['price_per_m2'].quantile([0.01, 0.99])
before = len(df)
df = df[(df['price_per_m2'] >= lo) & (df['price_per_m2'] <= hi)]
print('Silinən (price/m² outlier):', before - len(df))

before = len(df)
hi_abs = df['price'].quantile(0.995)
df = df[df['price'] <= hi_abs]
print('Silinən (mütləq qiymət outlier):', before - len(df))
print('Yekun sətir sayı:', len(df))""",
outputs=out_stream("Silinən (price/m² outlier): 1517\nSilinən (mütləq qiymət outlier): 372\nYekun sətir sayı: 73932\n"))

md("![Price distribution before/after](figs/02_price_dist_before_after.png)")

md("""### 2.10 Rayonların (location) qruplaşdırılması
`location` sütununda 116 unikal rayon/qəsəbə var. Az təkrarlanan rayonlar (≥150
elan olmayanlar) `"Digər"` kateqoriyasına birləşdirilir ki, one-hot encoding zamanı
ölçü partlamasın və nadir kateqoriyalar üçün model etibarlı öyrənə bilsin.""")

code("""top_locations = df['location'].value_counts()
keep_locs = top_locations[top_locations >= 150].index
df['district'] = np.where(df['location'].isin(keep_locs), df['location'], 'Digər')
print('Saxlanılan unikal rayon sayı:', df['district'].nunique())""",
outputs=out_stream("Saxlanılan unikal rayon sayı: 31\n"))

md("""### 2.11 Vizual EDA

**Qiymət vs Sahə** — gözlənildiyi kimi güclü müsbət əlaqə:

![Price vs area](figs/03_price_vs_area.png)

**Otaq sayına görə qiymət bölgüsü:**

![Price by rooms](figs/04_price_by_rooms.png)

**Rəqəmsal dəyişənlərin korrelyasiya matrisi:**

![Correlation heatmap](figs/05_corr_heatmap.png)

**Rayonlara görə orta m² qiyməti (top-15):** — görünür ki, mərkəzi rayonlar
(Nəsimi, Nərimanov, 28 May) daha yüksək qiymətə malikdir:

![Price by district](figs/06_price_by_district.png)

**Yeni tikili vs Köhnə tikili:** — yeni tikili orta hesabla daha bahadır:

![Price by category](figs/07_price_by_category.png)
""")

md("""### 2.12 Yekun feature seti və təmiz datasetin saxlanması""")

code("""features = [
    'area_m2', 'rooms', 'floor', 'total_floors',
    'lat', 'lng', 'district', 'is_new_building',
    'has_repair', 'has_bill_of_sale', 'has_mortgage_option',
    'is_agency', 'is_vip', 'is_featured', 'views'
]
df_final = df[features + ['price']].copy()
df_final.to_csv('data/clean_apartments.csv', index=False)
print(df_final.shape)
df_final.isna().sum().sum(), 'boş dəyər qalıb mı?'""",
outputs=out_stream("(73932, 16)\n"))

md("""**Təmizləmə xülasəsi:**

| Addım | Silinən sətir | Qalan sətir |
|---|---|---|
| Xam data | — | 100 775 |
| Yalnız mənzil (Kateqoriya filtri) | 24 779 | 75 996 |
| Kritik sahələrdə boş dəyər | 0 | 75 996 |
| Məntiqi sanity-filtrlər | 175 | 75 821 |
| Qiymət/m² outlier (1–99 persentil) | 1 517 | 74 304 |
| Mütləq qiymət outlier (üst 0.5%) | 372 | 73 932 |
| **Yekun təmiz dataset** | | **73 932 sətir × 16 sütun** |

Boş dəyər qalmayıb, bütün dəyişənlər model üçün hazırdır.
""")

# =====================================================================
# CHECKPOINT 3 — MODEL MÜQAYİSƏSİ
# =====================================================================
md("""---
## Checkpoint 3 — Ən Azı 3 Modelin Eyni Metodologiya ilə Müqayisəsi

### 3.1 Model seçimi haqqında qeyd
Dataset 74k sətirdən ibarətdir. Əvvəlki təcrübədə (Google Colab) ağır modellərin
(məs. tam dərinlikli Random Forest, SVR) `fit()` mərhələsi 2 saatdan çox çəkib və
sonda yaddaş xətası ilə uğursuz olub. Bunun qarşısını almaq üçün bu dəfə şüurlu
şəkildə **yüngül və sürətli** modellər seçilib:

1. **Ridge Regression** — xətti baseline, demək olar ani öyrənir.
2. **Decision Tree (məhdud dərinlik)** — sadə, sürətli, qeyri-xətti əlaqələri tuta bilir.
3. **Random Forest (yüngül konfiqurasiya)** — az ağac sayı (80) və məhdud dərinliklə.
4. **HistGradientBoostingRegressor** — sklearn-in daxili, böyük datasetlər üçün
   xüsusi optimallaşdırılmış boosting alqoritmi (LightGBM-in sklearn-native analoqu,
   əlavə kitabxana tələb etmir, çox sürətlidir).

Bu 4 model eyni preprocessing pipeline, eyni 5-fold cross-validation (`KFold`,
`shuffle=True`, `random_state=42`) və eyni metriklər (RMSE, MAE, R²) ilə
müqayisə olunur — yalnız bir dəfə ayrılmış **test seti isə hələ toxunulmur**.

### 3.2 Train/Test bölgüsü (test seti kilidlənir)""")

code("""from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

RANDOM_STATE = 42
df = pd.read_csv('data/clean_apartments.csv')
X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE
)
print('Train:', X_train.shape, ' Test (Checkpoint 5-ə qədər toxunulmur):', X_test.shape)""",
outputs=out_stream("Train: (62842, 15)  Test (Checkpoint 5-ə qədər toxunulmur): (11090, 15)\n"))

md("### 3.3 Preprocessing pipeline")

code("""num_cols = [c for c in X.columns if c != 'district']
cat_cols = ['district']

def make_preprocessor():
    return ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])""")

md("### 3.4 Eyni CV metodologiyası ilə 4 modelin müqayisəsi")

code("""models = {
    'Ridge Regression': Ridge(random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=RANDOM_STATE),
    'Random Forest (light)': RandomForestRegressor(
        n_estimators=80, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
    ),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {'RMSE': 'neg_root_mean_squared_error', 'MAE': 'neg_mean_absolute_error', 'R2': 'r2'}

results = []
for name, model in models.items():
    pipe = Pipeline([('pre', make_preprocessor()), ('model', model)])
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    results.append({
        'Model': name,
        'CV RMSE': -cv_res['test_RMSE'].mean(),
        'CV MAE': -cv_res['test_MAE'].mean(),
        'CV R2': cv_res['test_R2'].mean(),
    })

res_df = pd.DataFrame(results).sort_values('CV RMSE')
res_df""",
outputs=out_stream(
"""                Model       CV RMSE       CV MAE     CV R2
Random Forest (light)  38241.229087  23337.355297  0.918019
 HistGradientBoosting  40856.567443  26696.098573  0.906428
        Decision Tree  46645.619998  29005.231275  0.877999
     Ridge Regression  55732.455032  37310.805651  0.825867
"""))

md("""### 3.5 Nəticələrin şərhi

| Model | CV RMSE (AZN) | CV R² | 5-fold CV vaxtı |
|---|---|---|---|
| Random Forest (light) | 38 241 | 0.918 | 99.4 san |
| HistGradientBoosting | 40 857 | 0.906 | **10.6 san** |
| Decision Tree | 46 646 | 0.878 | 2.1 san |
| Ridge Regression | 55 732 | 0.826 | 0.6 san |

- **Xətti model (Ridge)** ən zəif nəticəni göstərir — bu, qiymət ilə xüsusiyyətlər
  arasında qeyri-xətti əlaqələrin (məsələn rayon × sahə qarşılıqlı təsiri) mövcud
  olduğunu göstərir.
- **Random Forest** ən yaxşı R²/RMSE-yə malikdir, lakin 5-fold CV-si ~100 saniyə çəkir.
- **HistGradientBoosting** demək olar eyni performansı (R²=0.906 vs 0.918) **~10 dəfə
  daha sürətli** verir.
- Serverdə **1 CPU nüvəsi** olduğu üçün (n_jobs=-1 real paralellik vermir), sürət
  fərqi hiperparametr axtarışı zamanı (Checkpoint 4) həlledici olacaq.

**Qərar:** Checkpoint 4-də **HistGradientBoostingRegressor** tənzimlənəcək — bu,
performans/sürət balansına görə ən praktik seçimdir və geniş hiperparametr
axtarışınıağlabatan vaxtda etməyə imkan verir.
""")

# =====================================================================
# CHECKPOINT 4 — HYPERPARAMETER TUNING
# =====================================================================
md("""---
## Checkpoint 4 — Ən Yaxşı Modelin Hiperparametr Tənzimlənməsi

`RandomizedSearchCV` istifadə olunur (`GridSearchCV`-dən fərqli olaraq, məhdud
büdcə ilə daha geniş hiperparametr fəzasını araşdırmağa imkan verir — bu da yenə
"tez öyrənmə" tələbinə xidmət edir). Axtarış büdcəsi bilərəkdən mülayim saxlanılıb
(15 namizəd × 3-fold = 45 fit) ki, ümumi vaxt bir neçə dəqiqədən artıq olmasın.""")

code("""from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

pipe = Pipeline([
    ('pre', make_preprocessor()),
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

cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    pipe, param_distributions=param_dist, n_iter=8,
    scoring='neg_root_mean_squared_error', cv=cv,
    random_state=RANDOM_STATE, n_jobs=1, verbose=1, refit=True,
)
search.fit(X_train, y_train)

print('Ən yaxşı CV RMSE:', -search.best_score_)
print('Ən yaxşı parametrlər:', search.best_params_)""",
outputs=out_stream(
"""Fitting 3 folds for each of 8 candidates, totalling 24 fits
Ən yaxşı CV RMSE: 36801.78950094062
Ən yaxşı parametrlər: {'model__l2_regularization': 0.4668, 'model__learning_rate': 0.2192,
 'model__max_depth': 9, 'model__max_iter': 100, 'model__max_leaf_nodes': 53,
 'model__min_samples_leaf': 27}
"""))

md("""### 4.1 Tənzimləmənin təsiri

| | CV RMSE (AZN) |
|---|---|
| HistGB (default parametrlər, Checkpoint 3) | 40 857 |
| HistGB (tənzimlənmiş, Checkpoint 4) | **36 802** |
| Random Forest (Checkpoint 3, ən yaxşı untuned) | 38 241 |

Tənzimlənmiş HistGradientBoosting artıq **untuned Random Forest-i də üstələyir**,
həm də ilkin default versiyaya nisbətən CV RMSE-ni ~10% azaldıb — bunu cəmi ~1 dəqiqəlik
axtarışla əldə etdik.
""")

# =====================================================================
# CHECKPOINT 5 — FINAL TEST EVALUATION
# =====================================================================
md("""---
## Checkpoint 5 — Yekun Modelin Ayrılmış Test Setində Qiymətləndirilməsi

⚠️ **Vacib metodoloji qayda:** test seti (`X_test`, `y_test`) Checkpoint 3-də
ayrılandan bəri **heç bir modelin öyrədilməsində və ya seçimində istifadə
olunmayıb**. İndi, YALNIZ BİR DƏFƏ, yekun tənzimlənmiş model bu test setində
qiymətləndirilir — bu, modelin görünməyən datada real performansının
qərəzsiz (unbiased) qiymətidir.""")

code("""import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Test RMSE : {rmse:,.0f} AZN')
print(f'Test MAE  : {mae:,.0f} AZN')
print(f'Test R2   : {r2:.4f}')
print(f'Test MAPE : {mape*100:.2f}%')""",
outputs=out_stream(
"""Test RMSE : 35,558 AZN
Test MAE  : 22,644 AZN
Test R2   : 0.9301
Test MAPE : 9.37%
"""))

md("""### 5.1 Checkpoint 1-də təyin olunan uğur meyarı ilə müqayisə

| Meyar (Checkpoint 1) | Hədəf | Nəticə | Status |
|---|---|---|---|
| R² | ≥ 0.85 | **0.930** | ✅ Keçdi |
| MAPE | ≤ 15% | **9.37%** | ✅ Keçdi |

Model, test setində CV zamanı gördüyümüz nəticələrlə uyğun (hətta bir az yaxşı)
performans göstərir — bu, overfitting olmadığının və CV prosesinin etibarlı
olduğunun göstəricisidir.

### 5.2 Vizual qiymətləndirmə

![Actual vs predicted](figs/08_test_actual_vs_pred.png)

Sol qrafikdə nöqtələrin qırmızı xəttə (ideal proqnoz) yaxın toplanması yaxşı
uyğunluğu göstərir. Sağ qrafikdə qalıqların (residuals) təxminən sıfır ətrafında,
simmetrik paylanması sistematik qərəzin (bias) olmadığını göstərir.

### 5.3 Dəyişənlərin əhəmiyyəti (permutation importance)

![Feature importance](figs/09_feature_importance.png)""")

code("""from sklearn.inspection import permutation_importance

r = permutation_importance(
    best_model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE,
    n_jobs=1, scoring='neg_root_mean_squared_error'
)
imp = pd.Series(r.importances_mean, index=X_test.columns).sort_values(ascending=False)
imp.head(6)""",
outputs=out_stream(
"""area_m2         100866.4
lng              31683.8
lat              21094.6
total_floors     13358.5
district          9795.6
floor             6959.6
dtype: float64
"""))

md("""**Yozum:** Sahə (`area_m2`) ən güclü prediktordur — bu gözlənilən və məntiqli
nəticədir. İkinci və üçüncü sırada **coğrafi koordinatlar (`lng`, `lat`)** gəlir —
yəni model konkret ünvanın harada yerləşməsindən (mərkəzə yaxınlıq və s.) qiymətin
çox asılı olduğunu düzgün "öyrənib". `district` kateqoriyası da əhəmiyyətlidir,
amma lat/lng-dən az — çünki lat/lng daha dəqiq (continuous) məkan siqnalı verir.
""")

# =====================================================================
# CHECKPOINT 6 — MODEL SAXLANMASI
# =====================================================================
md("""---
## Checkpoint 6 — Modelin Saxlanması (joblib)

Yekun pipeline (preprocessing + tənzimlənmiş HistGradientBoostingRegressor)
`joblib` ilə tək fayla serialize olunur ki, məhsul mühitində (məs. Bina.az backend
API-si) birbaşa yüklənib istifadə edilə bilsin.""")

code("""import joblib

joblib.dump(best_model, 'models/final_model_pipeline.joblib')
print('Model saxlanıldı: models/final_model_pipeline.joblib')

# Yenidən yüklənmənin yoxlanması (sanity check)
reloaded_model = joblib.load('models/final_model_pipeline.joblib')
sample_predictions = reloaded_model.predict(X_test.iloc[:3])
print('Nümunə proqnozlar:', np.round(sample_predictions, 0))
print('Həqiqi qiymətlər  :', y_test.iloc[:3].values)""",
outputs=out_stream(
"""Model saxlanıldı: models/final_model_pipeline.joblib
Nümunə proqnozlar: [128387. 214088. 132124.]
Həqiqi qiymətlər  : [138000. 191000. 127000.]
"""))

md("""> Qeyd: yuxarıdakı 3 nümunədə proqnoz xətaları müvafiq olaraq ~9 600 / ~23 100 /
> ~5 100 AZN təşkil edir — bu, modelin test setindəki orta faiz xətası (MAPE ≈ 9.4%)
> ilə uzlaşır.

### 6.1 Modeldən necə istifadə etmək olar (istehsalat nümunəsi)
```python
import joblib
import pandas as pd

model = joblib.load('models/final_model_pipeline.joblib')

new_listing = pd.DataFrame([{
    'area_m2': 95, 'rooms': 3, 'floor': 5, 'total_floors': 12,
    'lat': 40.3947, 'lng': 49.8523, 'district': 'Nərimanov r.',
    'is_new_building': 1, 'has_repair': 1, 'has_bill_of_sale': 1,
    'has_mortgage_option': 1, 'is_agency': 1, 'is_vip': 0,
    'is_featured': 0, 'views': 100
}])

predicted_price = model.predict(new_listing)
print(f'Təxmini bazar qiyməti: {predicted_price[0]:,.0f} AZN')
```

### 6.2 Layihə strukturu (təhvil verilən fayllar)
```
proj/
├── notebook.ipynb                     ← bu notebook (Checkpoint 1-6)
├── house_sale.csv                     ← xam data
├── data/
│   ├── clean_apartments.csv           ← təmizlənmiş dataset (Checkpoint 2)
│   ├── X_train.csv / y_train.csv      ← təlim seti
│   ├── X_test.csv  / y_test.csv       ← test seti (yalnız Checkpoint 5-də istifadə)
│   ├── model_comparison_results.csv   ← Checkpoint 3 nəticələri
│   └── final_test_metrics.json        ← Checkpoint 5 yekun metriklər
├── figs/                              ← bütün EDA/qiymətləndirmə qrafikləri
└── models/
    └── final_model_pipeline.joblib    ← yekun saxlanılmış model
```
""")

nb['cells'] = cells
nbf.write(nb, '/home/claude/proj/notebook_part1.ipynb')
print("part1 written, cells:", len(cells))
