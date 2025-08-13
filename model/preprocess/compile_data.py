import geopandas as gpd
import pandas as pd
from feature_engineering import add_feature_flags

# Constants
BUFFER_M = 10

# Load base data
points_gdf = gpd.read_file('model/datasets/sample/boston_sampled_points.geojson')
crashes_gdf = gpd.read_file('model/datasets/crash/pedestrian_crashes.geojson')

datasets = {
    'segments': gpd.read_file('model/datasets/features/boston_segments.geojson'),
    'ramps': gpd.read_file('model/datasets/features/pedestrian_ramp.geojson'),
    'crosswalk': gpd.read_file('model/datasets/features/sidewalk_centerline.geojson').query(
        "TYPE == 'CWALK-CL' or TYPE == 'CWALK-CL-UM'"
    ),
    'sidewalks': gpd.read_file('model/datasets/features/sidewalks.geojson'),
    'lamps': gpd.read_file('model/datasets/features/streetlight_locations.geojson'),
    'trees': gpd.read_file('model/datasets/features/trees_data.geojson')
}

print('data loaded')

# Reproject all
for key in ['segments','ramps','crosswalk','sidewalks','lamps','trees']:
    datasets[key] = datasets[key].to_crs(epsg=32619)
points_gdf = points_gdf.to_crs(epsg=32619)
crashes_gdf = crashes_gdf.to_crs(epsg=32619)

# Split day and night crashes
crashes_gdf['dispatch_ts'] = pd.to_datetime(crashes_gdf['dispatch_ts'])
crashes_gdf['hour'] = crashes_gdf['dispatch_ts'].dt.hour
day_crashes = crashes_gdf[crashes_gdf['hour'].between(6, 18)]
night_crashes = crashes_gdf[~crashes_gdf['hour'].between(6, 18)]

print('split success')

# Process day data boolean
day_points = add_feature_flags(points_gdf.copy(), day_crashes, datasets, BUFFER_M, False)
day_points.drop(columns='geometry').drop(columns='poor_lighting').to_csv(
    'model/datasets/final_csv/boolean_day_data.csv', index=False
)

print('boolean day data processed')

# Process day data continuous
day_points_c = add_feature_flags(points_gdf.copy(), day_crashes, datasets, BUFFER_M, True)
day_points_c.drop(columns='geometry').drop(columns='lighting').to_csv(
    'model/datasets/final_csv/continuous_day_data.csv', index=False
)

print('continuous day data processed')

# Process night data boolean
night_points = add_feature_flags(points_gdf.copy(), night_crashes, datasets, BUFFER_M, False)
night_points.drop(columns='geometry').to_csv('model/datasets/final_csv/boolean_night_data.csv', index=False)

print('boolean night data processed')

# Process night data continuous
night_points_c = add_feature_flags(points_gdf.copy(), night_crashes, datasets, BUFFER_M, True)
night_points_c.drop(columns='geometry').to_csv('model/datasets/final_csv/continuous_night_data.csv', index=False)

print('continuous night data processed')

# Process all data boolean
all_points = add_feature_flags(points_gdf.copy(), crashes_gdf, datasets, BUFFER_M, False)
all_points.drop(columns='geometry').to_csv('model/datasets/final_csv/boolean_all_data.csv', index=False)

print('boolean all data processed')

# Proccess all data continuous
all_points_c = add_feature_flags(points_gdf.copy(), crashes_gdf, datasets, BUFFER_M, True)
all_points_c.drop(columns='geometry').to_csv('model/datasets/final_csv/continuous_all_data.csv', index=False)

print('continuous all data processed')