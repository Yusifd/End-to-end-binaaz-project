import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 10

RAW_PATH = '/mnt/user-data/uploads/house_sale.csv'
CLEAN_PATH = '/home/claude/proj/data/clean_apartments.csv'
FIG = '/home/claude/proj/figs'

raw = pd.read_csv(RAW_PATH)
df = pd.read_csv(CLEAN_PATH)

# 1) Missingness in raw data (top columns with NaNs)
na = raw.isna().sum()
na = na[na > 0].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(na.index[::-1], (na.values[::-1] / len(raw) * 100), color='#4C72B0')
ax.set_xlabel('Boş dəyərlərin faizi (%)')
ax.set_title('Xam datada sütunlar üzrə boş (missing) dəyərlər')
plt.tight_layout()
plt.savefig(f'{FIG}/01_missing_raw.png')
plt.close()

# 2) Raw price distribution (log) vs cleaned price distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(raw['price'].clip(upper=raw['price'].quantile(0.99)), bins=60, color='#dd8452')
axes[0].set_title('Xam data: price (99-cu persentilə qədər)')
axes[0].set_xlabel('Qiymət (AZN)')
axes[1].hist(df['price'], bins=60, color='#55a868')
axes[1].set_title('Təmizlənmiş data: price')
axes[1].set_xlabel('Qiymət (AZN)')
plt.tight_layout()
plt.savefig(f'{FIG}/02_price_dist_before_after.png')
plt.close()

# 3) Price vs area scatter
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df['area_m2'], df['price'], s=4, alpha=0.25, color='#4C72B0')
ax.set_xlabel('Sahə (m²)')
ax.set_ylabel('Qiymət (AZN)')
ax.set_title('Qiymət vs Sahə')
plt.tight_layout()
plt.savefig(f'{FIG}/03_price_vs_area.png')
plt.close()

# 4) Boxplot price by rooms
fig, ax = plt.subplots(figsize=(8, 5))
order = sorted(df['rooms'].unique())
data = [df.loc[df['rooms'] == r, 'price'] for r in order]
ax.boxplot(data, labels=[int(r) for r in order], showfliers=False)
ax.set_xlabel('Otaq sayı')
ax.set_ylabel('Qiymət (AZN)')
ax.set_title('Otaq sayına görə qiymət bölgüsü')
plt.tight_layout()
plt.savefig(f'{FIG}/04_price_by_rooms.png')
plt.close()

# 5) Correlation heatmap
num_cols = ['area_m2', 'rooms', 'floor', 'total_floors', 'lat', 'lng', 'views',
            'is_new_building', 'has_repair', 'has_bill_of_sale', 'has_mortgage_option',
            'is_agency', 'is_vip', 'is_featured', 'price']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=90)
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=6)
fig.colorbar(im)
ax.set_title('Say. dəyişənlərin korrelyasiya matrisi')
plt.tight_layout()
plt.savefig(f'{FIG}/05_corr_heatmap.png')
plt.close()

# 6) Avg price per m2 by district (top 15 by count)
top_districts = df['district'].value_counts().head(15).index
d = df[df['district'].isin(top_districts)].copy()
d['ppm2'] = d['price'] / d['area_m2']
avg = d.groupby('district')['ppm2'].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(avg.index, avg.values, color='#8172B2')
ax.set_xlabel('Orta qiymət (AZN/m²)')
ax.set_title('Rayonlara görə orta m² qiyməti (top-15 rayon)')
plt.tight_layout()
plt.savefig(f'{FIG}/06_price_by_district.png')
plt.close()

# 7) New vs old building
fig, ax = plt.subplots(figsize=(6, 5))
data2 = [df.loc[df['is_new_building'] == 0, 'price'], df.loc[df['is_new_building'] == 1, 'price']]
ax.boxplot(data2, labels=['Köhnə tikili', 'Yeni tikili'], showfliers=False)
ax.set_ylabel('Qiymət (AZN)')
ax.set_title('Bina növünə görə qiymət')
plt.tight_layout()
plt.savefig(f'{FIG}/07_price_by_category.png')
plt.close()

print('EDA figures saved.')
print(df[num_cols].describe().T)
