"""CSC311 Fall 2026: reproducible training-only exploratory data analysis.

Run in the folder containing csc311_project_training_202609.csv.
Processed CSV files are exported to processed_data/; figures are displayed interactively.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif

DATA_FILE = "csc311_project_training_202609.csv"
SEED = 42

EASE_COL = ('How easy do you think it would be for someone with no prior experience '
            'to start playing this sport recreationally? Rate it on a scale from 0 to 10, '
            'where 0 means extremely difficult and 10 means extremely easy.')
PRICE_COL = ('How much would you be willing to pay for a decent seat '
             'at a professional competition of this sport?')
LIKERT_COLS = [
    'Success in this sport depends heavily on strategy and planning.',
    'Players must be comfortable with a high risk of injury.',
    'Individual brilliance can outweigh teamwork in this sport.'
]
MULTI_COLS = {
    'Where might you play this sport?': 'location',
    'During which seasons would you be most likely to play this sport recreationally in Toronto?': 'season',
    'What types of music do you most associate with this sport?': 'music'
}
TEXT_COLS = [
    'Which song do you associate with this sport?',
    'What foods and drinks do you associate with watching this sport in the place or culture you know best?',
    'What colour do you associate with this sport?',
    'How do you feel when you watch this sport?'
]
LIKERT_MAP = {'Strongly disagree': -2, 'Disagree': -1, 'Neutral': 0,
              'Agree': 1, 'Strongly agree': 2}
MISSING_MARKERS = {'', 'n/a', 'na', 'null', 'none', 'nan', '-', 'unknown'}


def normalize_missing(data):
    data = data.copy()
    for col in data.select_dtypes(include='object').columns:
        if col in {'id', 'label'}:
            continue
        series = data[col].astype('string').str.strip()
        data[col] = series.mask(series.str.lower().isin(MISSING_MARKERS))
    return data


def parse_price(value):
    """Parse simple monetary numbers/ranges; leave ambiguous language as missing.

    Currency conversion is NOT attempted (e.g. USD and CAD are not interchangeable).
    """
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {'free', 'nothing', 'zero', 'unwilling', '0', '$0'}:
        return 0.0
    if re.search(r'\b(usd|eur|gbp|cny|rmb|yuan|euro|pound|vnd)\b|[€£¥₫]', text):
        return np.nan
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    numbers = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', text)]
    if len(numbers) == 2 and re.search(r'\bto\b|[-–]', text):
        return sum(numbers) / 2
    if len(numbers) == 1:
        return numbers[0]
    return np.nan


def summarize_feature_types(data):
    print('\n=== FEATURE INVENTORY (RAW SURVEY) ===')
    for col in data.columns:
        if col in {'id', 'label'}:
            continue
        kind = ('numeric' if col in [EASE_COL, PRICE_COL] else
                'ordinal (Likert)' if col in LIKERT_COLS else
                'multi-select' if col in MULTI_COLS else 'free text')
        print(f'{kind:19} | {col}')


def print_numeric_distribution(data, col):
    nonmissing = data[col].dropna()
    if nonmissing.empty:
        print(f'{col}: no valid numeric responses')
        return
    print(f'\n{col}\n{nonmissing.describe(percentiles=[.25, .5, .75]).to_string()}')
    q1, q3 = nonmissing.quantile([.25, .75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = nonmissing[(nonmissing < low) | (nonmissing > high)]
    print(f'IQR flags: {len(flagged)}/{len(nonmissing)} outside [{low:.2f}, {high:.2f}]')
    print('IQR flags are diagnostic only; observations are NOT removed.')


# ==========================================
# PHASE 1: DATA LOADING & SPLITTING
# ==========================================
df = pd.read_csv(DATA_FILE)
unique_ids = df['id'].unique()
train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=SEED)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=SEED)
train_df = normalize_missing(df[df['id'].isin(train_ids)])
val_df = normalize_missing(df[df['id'].isin(val_ids)])
test_df = normalize_missing(df[df['id'].isin(test_ids)])

print(f'Dataset size: {len(df)} rows, {len(unique_ids)} respondents')
print(f'Split (respondents): train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}')
print(f'Split (rows): train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')
print('\n=== TRAIN CLASS COUNTS ===')
print(train_df['label'].value_counts().to_string())
summarize_feature_types(train_df)

# ==========================================
# PHASE 2: DATA QUALITY AND MISSING VALUES
# (All analyses/parameter estimates use training data only.)
# ==========================================
print('\n=== TRAIN MISSING PERCENTAGES (BEFORE IMPUTATION) ===')
print((train_df.drop(columns=['id', 'label']).isna().mean() * 100))
print('\n=== TRAIN MISSING PERCENTAGES BY CLASS ===')
print((train_df.assign(**{c: train_df[c].isna().astype(int)
                         for c in train_df.columns if c not in {'id', 'label'}})
       .groupby('label')[[c for c in train_df if c not in {'id', 'label'}]].mean() * 100)
      .round(1).T.to_string())

# Parse unstructured numeric responses and audit conversion outcomes BEFORE imputation.
for data in [train_df, val_df, test_df]:
    data[EASE_COL] = pd.to_numeric(data[EASE_COL], errors='coerce')
    data[EASE_COL] = data[EASE_COL].where(data[EASE_COL].between(0, 10))
    data[PRICE_COL] = data[PRICE_COL].apply(parse_price)
    data[PRICE_COL] = data[PRICE_COL].where(data[PRICE_COL] >= 0)

for col in [EASE_COL, PRICE_COL]:
    original_nonmissing = normalize_missing(df[df['id'].isin(train_ids)])[col].notna().sum()
    parsed_nonmissing = train_df[col].notna().sum()
    print(f'\nNumeric parsing: {col}\nOriginal nonmissing={original_nonmissing}, '
          f'parsed valid={parsed_nonmissing}, unparseable/invalid={original_nonmissing-parsed_nonmissing}')
    print_numeric_distribution(train_df, col)

# Audit a few unparsable original monetary responses, without examining validation/test.
raw_train_price = normalize_missing(df[df['id'].isin(train_ids)])[PRICE_COL]
unparsed_mask = raw_train_price.notna() & train_df[PRICE_COL].isna()
print('\n=== EXAMPLES OF UNPARSEABLE TRAIN PRICES (up to 12) ===')
print(raw_train_price[unparsed_mask].head(12).to_list())

for col in [EASE_COL, PRICE_COL]:
    for data in [train_df, val_df, test_df]:
        data[col + '_missing'] = data[col].isna().astype(int)
    median_val = train_df[col].median()
    if pd.isna(median_val):
        raise ValueError(f'No valid training observations for {col}')
    for data in [train_df, val_df, test_df]:
        data[col] = data[col].fillna(median_val)

# Missing free-text answers should contribute no TF-IDF tokens.
# Do this BEFORE filling remaining categorical missing values with 'Unknown'.
for data in [train_df, val_df, test_df]:
    for col in TEXT_COLS:
        data[col] = data[col].fillna('')
    data.fillna('Unknown', inplace=True)

# ==========================================
# PHASE 3: FEATURE ENGINEERING
# ==========================================
# 3.1 Centered ordinal Likert encoding (-2 to +2) with train-only median imputation.
for col in LIKERT_COLS:
    for data in [train_df, val_df, test_df]:
        data[col + '_missing'] = data[col].eq('Unknown').fillna(False).astype(int)
        data[col] = data[col].map(LIKERT_MAP)
    median_val = train_df[col].median()
    for data in [train_df, val_df, test_df]:
        data[col] = data[col].fillna(median_val)

# 3.2 Multi-hot encoding (including music types). Vocabulary comes from train.
for col, prefix in MULTI_COLS.items():
    encoded_sets = []
    for data in [train_df, val_df, test_df]:
        choices = data[col].astype('string').str.replace(r'\s*,\s*', ',', regex=True)
        encoded = choices.str.get_dummies(sep=',').drop(columns=['Unknown'], errors='ignore')
        encoded_sets.append(encoded)
        data[prefix + '_missing'] = data[col].eq('Unknown').fillna(False).astype(int)
    train_encoded, val_encoded, test_encoded = encoded_sets
    val_encoded = val_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    train_df = pd.concat([train_df.drop(columns=col), train_encoded.add_prefix(prefix + '_')], axis=1)
    val_df = pd.concat([val_df.drop(columns=col), val_encoded.add_prefix(prefix + '_')], axis=1)
    test_df = pd.concat([test_df.drop(columns=col), test_encoded.add_prefix(prefix + '_')], axis=1)

# 3.3 TF-IDF for the four free-text questions; missing answers are empty strings.
for col in TEXT_COLS:
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    train_matrix = tfidf.fit_transform(train_df[col].astype(str))
    names = [f'{col[:4]}_tfidf_{word}' for word in tfidf.get_feature_names_out()]
    train_features = pd.DataFrame(train_matrix.toarray(), columns=names, index=train_df.index)
    val_features = pd.DataFrame(tfidf.transform(val_df[col]).toarray(),
                                columns=names, index=val_df.index)
    test_features = pd.DataFrame(tfidf.transform(test_df[col]).toarray(),
                                 columns=names, index=test_df.index)
    train_df = pd.concat([train_df.drop(columns=col), train_features], axis=1)
    val_df = pd.concat([val_df.drop(columns=col), val_features], axis=1)
    test_df = pd.concat([test_df.drop(columns=col), test_features], axis=1)

# ==========================================
# PHASE 4: FEATURE SCALING AND VALIDATION
# ==========================================
X_train = train_df.drop(columns=['id', 'label'])
X_val = val_df.drop(columns=['id', 'label'])
X_test = test_df.drop(columns=['id', 'label'])
y_train, y_val, y_test = train_df['label'], val_df['label'], test_df['label']
scaler = StandardScaler()
X_train_scaled_df = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_val_scaled_df = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
X_test_scaled_df = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

assert 'id' not in X_train.columns
assert X_train.columns.equals(X_val.columns) and X_train.columns.equals(X_test.columns)
assert all(np.isfinite(X.to_numpy(dtype=float)).all() for X in [X_train, X_val, X_test])
assert set(train_ids).isdisjoint(val_ids) and set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)
print(f'\nPreprocessing checks passed; {X_train.shape[1]} model features.')

# Export model-ready datasets, preserving labels and consistent feature order.
# "clean" retains original numeric/Likert/multi-hot/TF-IDF scales;
# "scaled" uses training-fitted StandardScaler.
OUTPUT_DIR = Path('processed_data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for split_name, X_clean, X_scaled, y in [
    ('train', X_train, X_train_scaled_df, y_train),
    ('val', X_val, X_val_scaled_df, y_val),
    ('test', X_test, X_test_scaled_df, y_test),
]:
    for suffix, features in [('clean', X_clean), ('scaled', X_scaled)]:
        path = OUTPUT_DIR / f'{split_name}_{suffix}.csv'
        exported = features.copy()
        exported['label'] = y.reindex(exported.index)
        exported.to_csv(path, index=False)
        print(f'Exported {path} ({len(exported)} rows, {len(features.columns)} features + label)')


# ==========================================
# PHASE 5: TRAIN-ONLY EXPLORATORY ANALYSIS
# ==========================================
explore_df = X_train.copy()  # Unscaled values preserve interpretability.
explore_df['label'] = y_train

# 5.1 Numeric and Likert distributions, within and between classes.
print('\n=== NUMERIC/LIKERT FEATURES BY CLASS (MEAN AND STD) ===')
print(explore_df.groupby('label')[[EASE_COL, PRICE_COL] + LIKERT_COLS]
      .agg(['mean', 'std']).round(2).to_string())

# Use short subplot titles and axis labels to prevent overlapping text.
plot_features = [EASE_COL, PRICE_COL] + LIKERT_COLS
plot_titles = [
    'Ease (0–10)',
    'Ticket price',
    'Strategy and planning',
    'Injury risk',
    'Individual brilliance',
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
for ax, col, title in zip(axes.flat, plot_features, plot_titles):
    sns.boxplot(data=explore_df, x='label', y=col, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Sport class')
    ax.set_ylabel('Value' if col in (EASE_COL, PRICE_COL) else 'Likert score (-2 to +2)')
    ax.tick_params(axis='x', rotation=0)

# Remove the unused sixth subplot.
fig.delaxes(axes.flat[-1])
fig.suptitle('Numeric and Likert features by class (training data)', fontsize=16)
plt.show()

# The ticket price is right-skewed; log1p is for visualization only.
plt.figure(figsize=(7, 4))
sns.histplot(data=explore_df.assign(log_price=np.log1p(explore_df[PRICE_COL])),
             x='log_price', hue='label', element='step', stat='density', common_norm=False)
plt.title('Ticket-price distribution by class (log(1 + price))')
plt.xlabel('log(1 + parsed ticket price)')
plt.tight_layout()
plt.show()

# 5.2 Selection rates for multi-select questions (binary, UNscaled).
for prefix in ('season_', 'location_', 'music_'):
    cols = [c for c in X_train if c.startswith(prefix) and not c.endswith('_missing')]
    if cols:
        rate = explore_df.groupby('label')[cols].mean() * 100
        print(f'\n=== {prefix.upper()} SELECTION RATE (%) BY CLASS ===')
        print(rate.round(1).T.to_string())
        rate.T.plot(kind='bar', figsize=(10, 4), width=0.8)
        plt.title(f'{prefix[:-1].capitalize()} selection rates by class')
        plt.ylabel('Selection rate (%)')
        plt.xlabel('Choice')
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# 5.3 Pearson correlation among the five original numeric/Likert features.
correlation_cols = [EASE_COL, PRICE_COL] + LIKERT_COLS
corr = explore_df[correlation_cols].corr(method='spearman')
short_labels = ['Ease', 'Price', 'Strategy', 'Injury risk', 'Individual skill']
print('\n=== SPEARMAN CORRELATIONS (TRAIN) ===')
print(corr.set_axis(short_labels).set_axis(short_labels, axis=1).round(2).to_string())
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0,
            xticklabels=short_labels, yticklabels=short_labels, vmin=-1, vmax=1)
plt.title('Numeric/Likert feature correlation (train)')
plt.tight_layout()
plt.show()

# 5.4 Mean TF-IDF scores of free-text terms by sport (no class contrast).
tfidf_cols = [c for c in X_train if '_tfidf_' in c]
if tfidf_cols:
    mean_tfidf = explore_df.groupby('label')[tfidf_cols].mean()
    print('\n=== TOP 5 TERMS BY MEAN TF-IDF SCORE PER SPORT ===')
    for sport in mean_tfidf.index:
        print(f'\n{sport}:')
        for feature, score in mean_tfidf.loc[sport].nlargest(5).items():
            print(f'  {feature}: {score:.3f}')

# 5.5 Univariate feature-label associations; use train only.
X_eval = X_train.loc[:, X_train.nunique(dropna=False) > 1].copy()
discrete_mask = np.array([
    c.startswith(('season_', 'location_', 'music_')) or c.endswith('_missing')
    for c in X_eval.columns
])
f_scores, p_values = f_classif(X_eval, y_train)
mi_scores = mutual_info_classif(X_eval, y_train,
                                 discrete_features=discrete_mask,
                                 random_state=SEED)
stats_df = pd.DataFrame({
    'Feature': X_eval.columns,
    'ANOVA F-score': f_scores,
    'ANOVA p-value': p_values,
    'Mutual information': mi_scores,
}).sort_values('Mutual information', ascending=False)
print('\n=== TOP 15 FEATURES BY MUTUAL INFORMATION ===')
print(stats_df.head(15).round(4).to_string(index=False))
print('\n=== TOP 15 FEATURES BY ANOVA F-SCORE ===')
print(stats_df.sort_values('ANOVA F-score', ascending=False)
      .head(15).round(4).to_string(index=False))
print('\nExploration complete.')
