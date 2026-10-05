import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
import subprocess
import joblib

print('1. Loading data, cleaning, and engineering domain features...')

# 1. Load datasets
train_df = pd.read_csv('train-test.csv')
val_df = pd.read_csv('validation.csv')
dec_df = pd.read_csv('december-chart-inputs.csv')

# 2. Data Cleaning: Rectify negative weights and impute missing values
w_med = train_df['weight'].abs().median()
m_med = train_df['market_index'].median()

for df in [train_df, val_df, dec_df]:
    df['weight'] = df['weight'].abs().fillna(w_med)
    df['market_index'] = df.get('market_index', pd.Series()).fillna(m_med)

# 3. Enrich missing December route inputs (coordinates and market signals)
all_data = pd.concat([train_df, val_df])
p_coords = all_data.groupby('pickup')[['pickup_lat', 'pickup_lon']].mean()
d_coords = all_data.groupby('delivery')[['delivery_lat', 'delivery_lon']].mean()
dec_signals = val_df[val_df['date'].str.startswith('2025-12')].groupby('date')[['market_index', 'quote_signal']].mean()

dec_df['pickup_lat'] = dec_df['pickup'].map(p_coords['pickup_lat'])
dec_df['pickup_lon'] = dec_df['pickup'].map(p_coords['pickup_lon'])
dec_df['delivery_lat'] = dec_df['delivery'].map(d_coords['delivery_lat'])
dec_df['delivery_lon'] = dec_df['delivery'].map(d_coords['delivery_lon'])
dec_df['market_index'] = dec_df['date'].map(dec_signals['market_index']).fillna(m_med)
dec_df['quote_signal'] = dec_df['date'].map(dec_signals['quote_signal']).fillna(train_df['quote_signal'].median())

# 4. Feature Engineering: Temporal, Geospatial, and Economic Metrics
for df in [train_df, val_df, dec_df]:
    dt = pd.to_datetime(df['date'])
    df['month'], df['day'], df['dayofweek'], df['dayofyear'] = dt.dt.month, dt.dt.day, dt.dt.dayofweek, dt.dt.dayofyear
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    
    # Distance and coordinate metrics
    df['lat_diff'] = (df['delivery_lat'] - df['pickup_lat']).abs()
    df['lon_diff'] = (df['delivery_lon'] - df['pickup_lon']).abs()
    df['manhattan_dist'] = df['lat_diff'] + df['lon_diff']
    df['haversine_approx'] = np.sqrt(df['lat_diff']**2 + (df['lon_diff']*np.cos(np.radians(df['pickup_lat'])))**2) * 69.0
    
    # Financial and energy proxies
    df['estimated_base'] = df['distance'] * df['quote_signal']
    df['rate_per_mile_signal'] = df['quote_signal'] * df['market_index']
    df['weight_dist'] = df['weight'] * df['distance']
    df['weight_per_mile'] = df['weight'] / (df['distance'] + 1)
    df['log_distance'] = np.log1p(df['distance'])
    df['log_weight'] = np.log1p(df['weight'])

# 5. Categorical Encoding (Equipment, Pickup, Delivery)
cat_cols = ['equipment', 'pickup', 'delivery']
features = [c for c in train_df.columns if c not in ['load_id', 'posted_rate', 'date']]
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
all_cats = pd.concat([df[cat_cols].astype(str) for df in [train_df, val_df, dec_df]])
encoder.fit(all_cats)

for df in [train_df, val_df, dec_df]:
    df[cat_cols] = encoder.transform(df[cat_cols].astype(str))

cat_mask = [c in cat_cols for c in features]

# 6. Train Specialized Ensemble Models on 48,000 labeled records
print('2. Training specialized ensemble models with L1 loss...')

# Model 1: Rate-per-Mile specialist (optimizes rate/distance)
m_rpm = HistGradientBoostingRegressor(
    loss='absolute_error', 
    categorical_features=cat_mask, 
    max_iter=900, 
    learning_rate=0.03, 
    max_leaf_nodes=63, 
    min_samples_leaf=20, 
    l2_regularization=0.5, 
    random_state=42
)
m_rpm.fit(train_df[features], train_df['posted_rate'] / train_df['distance'])

# Model 2: Residual specialist (optimizes rate - estimated_base)
m_res = HistGradientBoostingRegressor(
    loss='absolute_error', 
    categorical_features=cat_mask, 
    max_iter=900, 
    learning_rate=0.03, 
    max_leaf_nodes=63, 
    min_samples_leaf=20, 
    l2_regularization=0.5, 
    random_state=42
)
m_res.fit(train_df[features], train_df['posted_rate'] - train_df['estimated_base'])

# Model 3: Direct baseline model
m_l1 = HistGradientBoostingRegressor(
    loss='absolute_error', 
    categorical_features=cat_mask, 
    max_iter=900, 
    learning_rate=0.03, 
    max_leaf_nodes=63, 
    min_samples_leaf=20, 
    l2_regularization=0.5, 
    random_state=42
)
m_l1.fit(train_df[features], train_df['posted_rate'])

# 7. Generate Validation Predictions (12,000 holdout loads)
print('3. Generating final validation predictions...')
p_val_rpm = m_rpm.predict(val_df[features]) * val_df['distance']
p_val_res = m_res.predict(val_df[features]) + val_df['estimated_base']
p_val_l1  = m_l1.predict(val_df[features])

val_preds = np.clip(0.55 * p_val_rpm + 0.35 * p_val_res + 0.10 * p_val_l1, 1.0, None)

pd.DataFrame({
    'load_id': val_df['load_id'], 
    'predicted_rate': np.round(val_preds, 2)
}).to_csv('validation_predictions.csv', index=False)
print('Saved validation_predictions.csv successfully!')

# 8. Generate December Benchmark Predictions (31 days)
print('4. Generating December route predictions...')
p_dec_rpm = m_rpm.predict(dec_df[features]) * dec_df['distance']
p_dec_res = m_res.predict(dec_df[features]) + dec_df['estimated_base']
p_dec_l1  = m_l1.predict(dec_df[features])

dec_preds = np.clip(0.55 * p_dec_rpm + 0.35 * p_dec_res + 0.10 * p_dec_l1, 1.0, None)

dec_orig = pd.read_csv('december-chart-inputs.csv')
dec_orig['predicted_rate'] = np.round(dec_preds, 2)
dec_orig[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']].to_csv('december-chart-inputs.csv', index=False)
print('Updated december-chart-inputs.csv successfully!')

# 9. Execute Official Scorer Script
print('\n5. Running score.py validation:')
subprocess.run([
    'python', 'score.py', 
    '--predictions', 'validation_predictions.csv', 
    '--december-predictions', 'december-chart-inputs.csv'
], check=True)

# 10. Persist Trained Model Artifacts
artifacts = {
    'm_rpm': m_rpm,
    'm_res': m_res,
    'm_l1': m_l1,
    'encoder': encoder,
    'features': features
}
joblib.dump(artifacts, 'spotter_freight_model.joblib')
print('Saved trained model artifacts to spotter_freight_model.joblib')
