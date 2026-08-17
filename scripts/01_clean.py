"""
Checkpoint 1-2: Data loading, cleaning, feature engineering.
Bina.az apartment (mənzil) price dataset.
"""
import pandas as pd
import numpy as np

RAW_PATH = '/mnt/user-data/uploads/house_sale.csv'
OUT_PATH = '/home/claude/proj/data/clean_apartments.csv'

def load_raw():
    df = pd.read_csv(RAW_PATH)
    return df

def clean(df):
    log = {}
    log['raw_rows'] = len(df)

    # ------------------------------------------------------------------
    # 1) SCOPE: keep only apartments (Yeni tikili / Köhnə tikili).
    #    Houses / land / offices / garages have fundamentally different
    #    pricing drivers (land area in "sot", building type, etc.) and
    #    mixing them with apartments would hurt model quality & the
    #    business question ("mənzil qiyməti proqnozu").
    # ------------------------------------------------------------------
    df = df[df['Kateqoriya'].isin(['Yeni tikili', 'Köhnə tikili'])].copy()
    log['after_category_filter'] = len(df)

    # ------------------------------------------------------------------
    # 2) Parse numeric fields hidden inside text columns
    # ------------------------------------------------------------------
    df['area_m2'] = df['Sahə'].str.extract(r'([\d.]+)').astype(float)

    floors = df['Mərtəbə'].str.extract(r'(\d+)\s*/\s*(\d+)')
    df['floor'] = pd.to_numeric(floors[0])
    df['total_floors'] = pd.to_numeric(floors[1])

    df['rooms'] = df['Otaq sayı']

    # ------------------------------------------------------------------
    # 3) Binary / categorical cleanup of "var/yoxdur"-style columns
    # ------------------------------------------------------------------
    df['has_repair'] = (df['Təmir'] == 'var').astype(int)
    df['has_bill_of_sale'] = (df['Çıxarış'] == 'var').astype(int)
    df['has_mortgage_option'] = (df['İpoteka'] == 'var').fillna(False).astype(int)
    df['is_agency'] = (df['owner_title'] == 'vasitəçi (agent)').astype(int)
    df['is_new_building'] = (df['Kateqoriya'] == 'Yeni tikili').astype(int)
    df['is_vip'] = df['vip'].notna().astype(int)
    df['is_featured'] = df['featured'].notna().astype(int)

    # ------------------------------------------------------------------
    # 4) Target column: price (== total_price for 99.99% of rows -> keep one)
    # ------------------------------------------------------------------
    df['price'] = df['price'].astype(float)

    # ------------------------------------------------------------------
    # 5) Drop rows with missing critical fields
    # ------------------------------------------------------------------
    critical = ['price', 'area_m2', 'floor', 'total_floors', 'rooms', 'lat', 'lng', 'location']
    before = len(df)
    df = df.dropna(subset=critical)
    log['dropped_missing_critical'] = before - len(df)

    # ------------------------------------------------------------------
    # 6) Logical sanity filters
    # ------------------------------------------------------------------
    before = len(df)
    df = df[df['floor'] <= df['total_floors']]
    df = df[(df['area_m2'] >= 15) & (df['area_m2'] <= 500)]
    df = df[(df['rooms'] >= 1) & (df['rooms'] <= 8)]
    log['dropped_logical_sanity'] = before - len(df)

    # ------------------------------------------------------------------
    # 7) Outlier removal on price / price-per-m2 (data entry errors,
    #    e.g. a 200m2 apartment listed at 600,000,000 AZN or 11 AZN)
    #    -> clip to the 1st-99th percentile of price-per-m2
    # ------------------------------------------------------------------
    df['price_per_m2'] = df['price'] / df['area_m2']
    lo, hi = df['price_per_m2'].quantile([0.01, 0.99])
    before = len(df)
    df = df[(df['price_per_m2'] >= lo) & (df['price_per_m2'] <= hi)]
    log['dropped_price_outliers'] = before - len(df)

    # Also drop absolute price outliers beyond 99.5th percentile (rare mansions
    # that would dominate the loss function without adding learnable signal)
    before = len(df)
    hi_abs = df['price'].quantile(0.995)
    df = df[df['price'] <= hi_abs]
    log['dropped_extreme_absolute_price'] = before - len(df)

    # ------------------------------------------------------------------
    # 8) Group rare districts (location) into "other" to control cardinality
    # ------------------------------------------------------------------
    top_locations = df['location'].value_counts()
    keep_locs = top_locations[top_locations >= 150].index
    df['district'] = np.where(df['location'].isin(keep_locs), df['location'], 'Digər')

    # ------------------------------------------------------------------
    # 9) Final feature set
    # ------------------------------------------------------------------
    features = [
        'area_m2', 'rooms', 'floor', 'total_floors',
        'lat', 'lng', 'district', 'is_new_building',
        'has_repair', 'has_bill_of_sale', 'has_mortgage_option',
        'is_agency', 'is_vip', 'is_featured', 'views'
    ]
    keep_cols = features + ['price']
    df_final = df[keep_cols].copy()

    log['final_rows'] = len(df_final)
    log['final_cols'] = df_final.shape[1]
    return df_final, log


if __name__ == '__main__':
    raw = load_raw()
    clean_df, log = clean(raw)
    clean_df.to_csv(OUT_PATH, index=False)
    print(clean_df.shape)
    print(clean_df.dtypes)
    print(clean_df.isna().sum())
    import json
    print(json.dumps(log, indent=2, ensure_ascii=False))
