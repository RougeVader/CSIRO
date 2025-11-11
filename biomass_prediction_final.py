"""
This script generates the final biomass prediction submission file.
This version incorporates historical weather data as features (X-Factor Phase 2).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing import image

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder


def image_batch_generator(image_paths, batch_size, preproc_func):
    """
    Yields batches of preprocessed images from a list of paths.
    """
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for img_path in batch_paths:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            batch_images.append(x)
        yield preproc_func(np.array(batch_images))

def predict_metadata_in_batch(image_paths, metadata_model, state_encoder, species_encoder, batch_size=32):
    """
    Predicts 'State' and 'Species' for a list of image paths using batch processing.
    """
    print("Predicting metadata for test images in batches...")
    preproc_func = lambda x: x / 255.0
    img_generator = image_batch_generator(image_paths, batch_size, preproc_func)
    
    all_state_preds, all_species_preds = [], []
    for batch in tqdm(img_generator, total=-(-len(image_paths) // batch_size)):
        state_pred, species_pred = metadata_model.predict(batch, verbose=0)
        all_state_preds.extend(np.argmax(state_pred, axis=1))
        all_species_preds.extend(np.argmax(species_pred, axis=1))
        
    states = state_encoder.inverse_transform(all_state_preds)
    species = species_encoder.inverse_transform(all_species_preds)
    return pd.DataFrame({'State': states, 'Species': species})

def extract_features_in_batch(image_paths, feature_extractor, batch_size=32):
    """
    Extracts image features for a list of image paths using batch processing.
    """
    print("Extracting EfficientNetB0 features in batches...")
    preproc_func = preprocess_input
    img_generator = image_batch_generator(image_paths, batch_size, preproc_func)
    
    all_features = [
        feature_extractor.predict(batch, verbose=0).reshape(batch.shape[0], -1)
        for batch in tqdm(img_generator, total=-(-len(image_paths) // batch_size))
    ]
    return np.vstack(all_features)

def generate_weather_features(df, weather_df, state_map, date_col='Sampling_Date'):
    """
    Generates historical weather features for a given dataframe.
    """
    print(f"Generating weather features for {len(df)} images...")
    all_features = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        state = row['State']
        location = state_map.get(state)
        if not location:
            all_features.append({'image_path': row['image_path']}) # Append empty features if no location
            continue
            
        sampling_date = row[date_col]
        start_date = sampling_date - pd.Timedelta(days=30)
        
        mask = (weather_df['Location'] == location) & (weather_df['Date'] >= start_date) & (weather_df['Date'] < sampling_date)
        past_weather = weather_df.loc[mask]
        
        features = {
            'image_path': row['image_path'],
            'total_rainfall_last_30d': past_weather['Rainfall'].sum(),
            'avg_min_temp_last_30d': past_weather['MinTemp'].mean(),
            'avg_max_temp_last_30d': past_weather['MaxTemp'].mean(),
            'avg_sunshine_last_30d': past_weather['Sunshine'].mean(),
            'days_with_rain_last_30d': (past_weather['Rainfall'] > 1).sum(),
        }
        all_features.append(features)
        
    return pd.DataFrame(all_features).fillna(0)

def main():
    """Main function to run the entire prediction and submission pipeline."""
    # 1. Load data, models, and mappings
    print("Loading data, models, and mappings...")
    data_path = Path('.')
    train_df = pd.read_csv(data_path / 'train.csv')
    test_df = pd.read_csv(data_path / 'test.csv')
    
    metadata_model = load_model(data_path / 'metadata_model.h5')
    with open(data_path / 'state_encoder.pkl', 'rb') as f:
        state_encoder = pickle.load(f)
    with open(data_path / 'species_encoder.pkl', 'rb') as f:
        species_encoder = pickle.load(f)
    with open(data_path / 'state_location_map.pkl', 'rb') as f:
        state_location_map = pickle.load(f)

    # Load and preprocess weather data once
    weather_df = pd.read_csv(data_path / 'weatherAUS.csv', parse_dates=['Date'])
    weather_df.sort_values(by=['Location', 'Date'], inplace=True)
    weather_cols_to_impute = ['Rainfall', 'MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'WindGustSpeed']
    weather_df[weather_cols_to_impute] = weather_df.groupby('Location')[weather_cols_to_impute].transform(lambda x: x.ffill().bfill())
    weather_df.fillna(0, inplace=True)

    # 2. Predict metadata for the test set
    test_image_paths = [data_path / p for p in test_df['image_path']]
    predicted_metadata = predict_metadata_in_batch(test_image_paths, metadata_model, state_encoder, species_encoder)
    test_df[['State', 'Species']] = predicted_metadata

    # 3. Prepare and pivot training data
    train_pivot = train_df.pivot_table(index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], columns='target_name', values='target').reset_index()
    train_pivot['Sampling_Date'] = pd.to_datetime(train_pivot['Sampling_Date'])
    
    # Load pre-computed training weather features
    train_weather_features = pd.read_csv(data_path / 'train_weather_features.csv')
    train_pivot = pd.merge(train_pivot, train_weather_features, on='image_path', how='left').fillna(0)
    
    train_pivot['month'] = train_pivot['Sampling_Date'].dt.month
    train_pivot['year'] = train_pivot['Sampling_Date'].dt.year
    train_pivot.drop('Sampling_Date', axis=1, inplace=True)

    # 4. Prepare test data
    test_pivot = test_df[['image_path', 'State', 'Species']].drop_duplicates().reset_index(drop=True)
    test_pivot['Pre_GSHH_NDVI'] = train_pivot['Pre_GSHH_NDVI'].mean()
    test_pivot['Height_Ave_cm'] = train_pivot['Height_Ave_cm'].mean()
    test_pivot['month'] = train_pivot['month'].mode()[0]
    test_pivot['year'] = train_pivot['year'].mode()[0]
    test_pivot['Sampling_Date'] = pd.to_datetime(test_pivot['year'].astype(str) + '-' + test_pivot['month'].astype(str) + '-15')
    
    # Generate test weather features on the fly
    test_weather_features = generate_weather_features(test_pivot, weather_df, state_location_map)
    test_pivot = pd.merge(test_pivot, test_weather_features, on='image_path', how='left').fillna(0)
    test_pivot.drop('Sampling_Date', axis=1, inplace=True)

    # 5. Extract image features
    feature_extractor = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    train_image_paths = [data_path / p for p in train_pivot['image_path']]
    test_image_paths = [data_path / p for p in test_pivot['image_path']]
    X_train_img = extract_features_in_batch(train_image_paths, feature_extractor)
    X_test_img = extract_features_in_batch(test_image_paths, feature_extractor)

    # 6. Prepare tabular data for ML model
    y_train = train_pivot[['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']]
    drop_cols_train = ['image_path', 'Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    X_train_tabular = train_pivot.drop(columns=drop_cols_train)
    X_test_tabular = test_pivot.drop(columns=['image_path'])

    # ** FIX: Align columns directly after creation to prevent key errors **
    X_test_tabular = X_test_tabular.reindex(columns=X_train_tabular.columns, fill_value=0)

    categorical_features = ['State', 'Species']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_encoded = encoder.fit_transform(X_train_tabular[categorical_features])
    X_test_encoded = encoder.transform(X_test_tabular[categorical_features])

    X_train_numerical = X_train_tabular.drop(columns=categorical_features).astype(np.float32)
    X_test_numerical = X_test_tabular.drop(columns=categorical_features).astype(np.float32)

    X_train_final = np.hstack([X_train_numerical.values, X_train_encoded, X_train_img])
    X_test_final = np.hstack([X_test_numerical.values, X_test_encoded, X_test_img])
    
    # 7. Train models and predict (Hierarchical approach)
    print("Training models with hierarchical strategy (X-Factor Phase 3)...")
    total_g_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    y_total_g = y_train['Dry_Total_g']
    total_g_model.fit(X_train_final, y_total_g)

    dry_total_g_pred_test = total_g_model.predict(X_test_final)
    dry_total_g_pred_train = cross_val_predict(total_g_model, X_train_final, y_total_g, cv=5, n_jobs=-1)

    X_train_augmented = np.hstack([X_train_final, dry_total_g_pred_train.reshape(-1, 1)])
    X_test_augmented = np.hstack([X_test_final, dry_total_g_pred_test.reshape(-1, 1)])

    component_targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'GDM_g']
    predictions = {'Dry_Total_g': dry_total_g_pred_test}

    for target in component_targets:
        print(f'-- Training model for {target}')
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_augmented, y_train[target])
        predictions[target] = model.predict(X_test_augmented)

    # 8. Post-processing and Submission
    print("Applying post-processing and creating submission file...")
    for target in predictions:
        predictions[target] = np.maximum(0, predictions[target])

    components = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g']
    component_sum = np.sum([predictions[c] for c in components], axis=0)
    scale_factor = np.divide(predictions['Dry_Total_g'], component_sum, out=np.ones_like(predictions['Dry_Total_g']), where=component_sum!=0)

    for comp in components:
        predictions[comp] *= scale_factor

    test_pivot['image_id'] = test_pivot['image_path'].apply(lambda x: Path(x).stem)
    submission_list = []
    for i, row in test_pivot.iterrows():
        for target_name in y_train.columns:
            submission_list.append({
                'sample_id': f"{row['image_id']}__{target_name}",
                'target': predictions[target_name][i]
            })

    submission_df = pd.DataFrame(submission_list)
    submission_df.to_csv('submission.csv', index=False)

    print('X-Factor Phase 3 submission file created successfully!')
    print(submission_df.head())

if __name__ == '__main__':
    main()