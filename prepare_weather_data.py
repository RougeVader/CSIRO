"""
This script prepares historical weather features for the CSIRO Biomass competition.

It performs the following steps:
1.  Defines coordinates for Australian state capitals and the weather stations 
    listed in the 'weatherAUS.csv' dataset.
2.  Calculates the nearest weather station for each state and saves this
    mapping to 'state_location_map.pkl'.
3.  Loads the 'weatherAUS.csv' dataset.
4.  Performs imputation for missing weather data (e.g., Rainfall, Temperature).
5.  For each unique image in the training set, it calculates aggregate weather
    features for the 30 days prior to the image's sampling date.
6.  Saves the generated features to 'train_weather_features.csv'.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm
import pickle

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees).
    """
    R = 6371  # Radius of Earth in kilometers
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def create_and_save_mapping(data_path):
    """
    Creates and saves a mapping from state to the nearest weather station.
    """
    state_capitals = {
        'NSW': {'lat': -33.8688, 'lon': 151.2093}, 'VIC': {'lat': -37.81417, 'lon': 144.96306},
        'QLD': {'lat': -27.46778, 'lon': 153.02806}, 'SA': {'lat': -34.921230, 'lon': 138.599503},
        'WA': {'lat': -31.9558, 'lon': 115.8597}, 'TAS': {'lat': -42.88056, 'lon': 147.32500},
    }
    
    weather_locations = {
        'Albury': {'lat': -36.07, 'lon': 146.92}, 'BadgerysCreek': {'lat': -33.88, 'lon': 150.75},
        'Cobar': {'lat': -31.50, 'lon': 145.83}, 'CoffsHarbour': {'lat': -30.30, 'lon': 153.11},
        'Moree': {'lat': -29.46, 'lon': 149.85}, 'Newcastle': {'lat': -32.92, 'lon': 151.78},
        'NorahHead': {'lat': -33.28, 'lon': 151.57}, 'NorfolkIsland': {'lat': -29.05, 'lon': 167.96},
        'Penrith': {'lat': -33.75, 'lon': 150.69}, 'Richmond': {'lat': -33.60, 'lon': 150.75},
        'Sydney': {'lat': -33.86, 'lon': 151.21}, 'SydneyAirport': {'lat': -33.95, 'lon': 151.18},
        'WaggaWagga': {'lat': -35.12, 'lon': 147.37}, 'Williamtown': {'lat': -32.81, 'lon': 151.84},
        'Wollongong': {'lat': -34.42, 'lon': 150.89}, 'Canberra': {'lat': -35.28, 'lon': 149.13},
        'Tuggeranong': {'lat': -35.42, 'lon': 149.09}, 'MountGinini': {'lat': -35.53, 'lon': 148.77},
        'Ballarat': {'lat': -37.55, 'lon': 143.85}, 'Bendigo': {'lat': -36.76, 'lon': 144.28},
        'Sale': {'lat': -38.11, 'lon': 147.06}, 'MelbourneAirport': {'lat': -37.67, 'lon': 144.84},
        'Melbourne': {'lat': -37.81, 'lon': 144.96}, 'Mildura': {'lat': -34.19, 'lon': 142.16},
        'Nhil': {'lat': -36.33, 'lon': 141.65}, 'Portland': {'lat': -38.35, 'lon': 141.60},
        'Watsonia': {'lat': -37.71, 'lon': 145.08}, 'Dartmoor': {'lat': -37.92, 'lon': 141.28},
        'Brisbane': {'lat': -27.47, 'lon': 153.03}, 'Cairns': {'lat': -16.92, 'lon': 145.77},
        'GoldCoast': {'lat': -28.00, 'lon': 153.43}, 'Townsville': {'lat': -19.26, 'lon': 146.82},
        'Adelaide': {'lat': -34.93, 'lon': 138.60}, 'MountGambier': {'lat': -37.82, 'lon': 140.78},
        'Nuriootpa': {'lat': -34.47, 'lon': 138.99}, 'Woomera': {'lat': -31.20, 'lon': 136.83},
        'Albany': {'lat': -35.02, 'lon': 117.88}, 'Witchcliffe': {'lat': -34.03, 'lon': 115.10},
        'PearceRAAF': {'lat': -31.67, 'lon': 116.01}, 'PerthAirport': {'lat': -31.94, 'lon': 115.97},
        'Perth': {'lat': -31.95, 'lon': 115.86}, 'SalmonGums': {'lat': -32.98, 'lon': 121.64},
        'Walpole': {'lat': -34.98, 'lon': 116.73}, 'Hobart': {'lat': -42.88, 'lon': 147.33},
        'Launceston': {'lat': -41.44, 'lon': 147.14}, 'AliceSprings': {'lat': -23.70, 'lon': 133.88},
        'Darwin': {'lat': -12.46, 'lon': 130.84}, 'Katherine': {'lat': -14.46, 'lon': 132.26},
        'Uluru': {'lat': -25.34, 'lon': 131.04}
    }

    mapping = {}
    for state, s_coords in state_capitals.items():
        min_dist = float('inf')
        best_location = None
        for loc, w_coords in weather_locations.items():
            dist = haversine_distance(s_coords['lat'], s_coords['lon'], w_coords['lat'], w_coords['lon'])
            if dist < min_dist:
                min_dist = dist
                best_location = loc
        mapping[state] = best_location
    
    print("Created State -> Location Mapping:")
    print(mapping)
    
    map_path = data_path / 'state_location_map.pkl'
    with open(map_path, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"Mapping saved to {map_path}")
    return mapping

def prepare_train_weather_features(data_path, state_location_map):
    """
    Loads weather and training data, calculates historical aggregate 
    weather features for the training set, and saves them to a CSV file.
    """
    print("Loading weatherAUS.csv...")
    weather_df = pd.read_csv(data_path / 'weatherAUS.csv', parse_dates=['Date'])
    
    # Impute missing numerical data by forward/backward filling within each location
    weather_df.sort_values(by=['Location', 'Date'], inplace=True)
    weather_cols_to_impute = ['Rainfall', 'MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'WindGustSpeed']
    print("Imputing missing weather data...")
    weather_df[weather_cols_to_impute] = weather_df.groupby('Location')[weather_cols_to_impute].transform(lambda x: x.ffill().bfill())
    weather_df[weather_cols_to_impute] = weather_df[weather_cols_to_impute].fillna(0)

    train_df = pd.read_csv(data_path / 'train.csv')[['image_path', 'Sampling_Date', 'State']].drop_duplicates()
    train_df['Sampling_Date'] = pd.to_datetime(train_df['Sampling_Date'])
    
    print(f"Processing {len(train_df)} unique training images to generate weather features...")
    all_features = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        state = row['State']
        location = state_location_map.get(state)
        if not location:
            continue
        
        sampling_date = row['Sampling_Date']
        start_date = sampling_date - pd.Timedelta(days=30)
        
        mask = (weather_df['Location'] == location) & (weather_df['Date'] >= start_date) & (weather_df['Date'] < sampling_date)
        past_weather = weather_df.loc[mask]
        
        features = {
            'image_path': row['image_path'],
            'total_rainfall_last_30d': past_weather['Rainfall'].sum(),
            'avg_min_temp_last_30d': past_weather['MinTemp'].mean(),
            'avg_max_temp_last_30d': past_weather['MaxTemp'].mean(),
            'avg_sunshine_last_30d': past_weather['Sunshine'].mean(),
            'days_with_rain_last_30d': (past_weather['Rainfall'] > 1).sum(), # Days with more than 1mm of rain
        }
        all_features.append(features)
        
    feature_df = pd.DataFrame(all_features).fillna(0)
    
    save_path = data_path / 'train_weather_features.csv'
    print(f"Saving training weather features to {save_path}")
    feature_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    DATA_PATH = Path('.')
    mapping = create_and_save_mapping(DATA_PATH)
    prepare_train_weather_features(DATA_PATH, mapping)
    print("Weather feature preparation for training data complete.")
