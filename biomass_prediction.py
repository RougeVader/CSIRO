import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data_path = Path('C:/Users/Ayush/OneDrive/Desktop/kaggle/New folder')
train_df = pd.read_csv(data_path / 'train.csv')
test_df = pd.read_csv(data_path / 'test.csv')

print('train_df shape:', train_df.shape)
print('test_df shape:', test_df.shape)
print(train_df.head())

train_pivot = train_df.pivot_table(index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], columns='target_name', values='target').reset_index()
print(train_pivot.head())

train_pivot['Sampling_Date'] = pd.to_datetime(train_pivot['Sampling_Date'])
train_pivot['month'] = train_pivot['Sampling_Date'].dt.month
train_pivot['year'] = train_pivot['Sampling_Date'].dt.year
train_pivot = train_pivot.drop('Sampling_Date', axis=1)

test_df['image_id'] = test_df['image_path'].apply(lambda x: x.split('/')[1].replace('.jpg', ''))

test_pivot = pd.DataFrame(test_df['image_path'].unique(), columns=['image_path'])
test_pivot['image_id'] = test_pivot['image_path'].apply(lambda x: x.split('/')[1].replace('.jpg', ''))

metadata = train_df[['State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']].drop_duplicates()

test_pivot['State'] = train_pivot['State'].mode()[0]
test_pivot['Species'] = train_pivot['Species'].mode()[0]
test_pivot['Pre_GSHH_NDVI'] = train_pivot['Pre_GSHH_NDVI'].mean()
test_pivot['Height_Ave_cm'] = train_pivot['Height_Ave_cm'].mean()
test_pivot['month'] = train_pivot['month'].mode()[0]
test_pivot['year'] = train_pivot['year'].mode()[0]

print(train_pivot.head())

X = train_pivot.drop(['image_path', 'Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g'], axis=1)
y = train_pivot[['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']]

X_test = test_pivot.drop(['image_path', 'image_id'], axis=1)

categorical_features = ['State', 'Species']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

X_numerical = X.drop(categorical_features, axis=1)
X_test_numerical = X_test.drop(categorical_features, axis=1)

X_final = np.hstack([X_numerical.values, X_encoded])
X_test_final = np.hstack([X_test_numerical.values, X_test_encoded])

models = {}
for target in y.columns:
    print(f'Training model for {target}')
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_final, y[target])
    models[target] = model

predictions = {}
for target, model in models.items():
    predictions[target] = model.predict(X_test_final)

submission_list = []
for i, row in test_pivot.iterrows():
    image_id = row['image_id']
    for target_name in y.columns:
        sample_id = f"{image_id}__{target_name}"
        prediction = predictions[target_name][i]
        submission_list.append({'sample_id': sample_id, 'target': prediction})

submission_df = pd.DataFrame(submission_list)
submission_df.to_csv(data_path /'submission.csv', index=False)

print('Submission file created successfully!')
print(submission_df.head())
