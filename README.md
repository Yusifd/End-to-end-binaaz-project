# End-to-end-binaaz-project

# 🏠 Bina.az Mənzil Qiymətinin Proqnozlaşdırılması

Bakıdakı mənzil elanlarının (Bina.az) xüsusiyyətlərinə (sahə, otaq sayı, rayon,
mərtəbə, təmir vəziyyəti və s.) əsasən bazar qiymətini proqnozlaşdıran uçdan-uca
maşın öyrənməsi layihəsi — xam datadan tutmuş, model müqayisəsinə və biznes
hesabatına qədər.

## 📥 Dataset

Bu layihə Bina.az-dan toplanmış mənzil elanları datasetindən istifadə edir
(100 775 sətir × 51 sütun).

**Dataset linki (Kaggle):**
👉 https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project

Layihəni sıfırdan işə salmaq üçün:
1. Yuxarıdakı linkdən `house_sale.csv` faylını endirin.
2. Faylı layihənin kök qovluğuna (bu README ilə eyni səviyyəyə) yerləşdirin.
3. `notebooks/Bina_Qiymet_Proqnozu.ipynb` faylını açıb başdan sona işə salın.

## 🎯 Problemin qısa təsviri

| | |
|---|---|
| **Problem növü** | Regressiya (kəsilməz qiymət proqnozu) |
| **Target dəyişən** | `price` (AZN) |
| **Əhatə dairəsi** | Bakı, mənzil elanları (Yeni tikili + Köhnə tikili) |
| **Uğur meyarı** | R² ≥ 0.85, MAPE ≤ 15% |

## 🔬 Metodologiya

1. **EDA + Təmizləmə** — 100 775 xam sətirdən 73 932 təmiz sətir (dublikat/ekvivalent
   sütunların silinməsi, mətn daxilindəki ədədi dəyərlərin çıxarılması, məntiqi
   sanity-filtrlər, qiymət outlier-lərinin persentil əsaslı təmizlənməsi).
2. **Model müqayisəsi** — 4 model (Ridge, Decision Tree, Random Forest, HistGradientBoosting),
   eyni 5-fold cross-validation, eyni metriklər (RMSE, MAE, R²).
3. **Hiperparametr tənzimləməsi** — ən yaxşı model üzərində `RandomizedSearchCV`.
4. **Yekun test qiymətləndirməsi** — ayrılmış test setində, YALNIZ BİR DƏFƏ.
5. **Modelin saxlanması** — `joblib` ilə.
6. **Biznes hesabatı** — qeyri-texniki, 2 səhifəlik icmal (`reports/`).

## 📊 Yekun nəticələr (test set, n=11 090)

| Metrika | Nəticə |
|---|---|
| RMSE | 35 558 AZN |
| MAE | 22 644 AZN |
| **R²** | **0.930** |
| **MAPE** | **9.37%** |

Model seçimi: **HistGradientBoostingRegressor** (tənzimlənmiş) — performans/sürət
balansına görə seçildi (5-fold CV cəmi ~10 saniyə, Random Forest-ə yaxın dəqiqliklə).

## 📁 Repo strukturu

```
.
├── notebooks/
│   └── Bina_Qiymet_Proqnozu.ipynb      # Tam pipeline: EDA → model → tuning → test
├── scripts/                             # Mərhələ-mərhələ .py skriptlər
│   ├── 01_clean.py
│   ├── 02_eda.py
│   ├── 03_model_comparison.py
│   ├── 04_tuning.py
│   └── 05_final_eval_and_save.py
├── data/
│   ├── clean_apartments.csv             # Təmizlənmiş dataset
│   ├── model_comparison_results.csv     # Checkpoint 3 nəticələri
│   └── final_test_metrics.json          # Checkpoint 5 yekun metriklər
├── figs/                                # EDA və qiymətləndirmə qrafikləri (9 ədəd)
├── models/
│   └── final_model_pipeline.joblib      # Yekun saxlanılmış model
├── reports/
│   └── Biznes_Hesabati.docx             # Qeyri-texniki biznes hesabatı
├── requirements.txt
└── README.md
```

> ⚠️ Xam `house_sale.csv` GitHub-un 100MB fayl limitini keçdiyi üçün repoda yoxdur —
> yuxarıdakı Kaggle linkindən endirin.

## ⚙️ Quraşdırma və işə salma

```bash
git clone https://github.com/<istifadəçi-adınız>/<repo-adı>.git
cd <repo-adı>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# house_sale.csv-ni Kaggle-dan endirib kök qovluğa qoyun, sonra:
jupyter notebook notebooks/Bina_Qiymet_Proqnozu.ipynb
```

## 🚀 Saxlanılmış modeldən istifadə

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

## 🛠️ İstifadə olunan texnologiyalar

Python · pandas · NumPy · scikit-learn · matplotlib · joblib · scipy
