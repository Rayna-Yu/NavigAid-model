import pandas as pd
import json

# Load crash CSV file
df = pd.read_csv('model/datasets/crash/boston_crash.csv')

# Filter pedestrian crashes and drop rows without coordinates
ped_df = df[df['mode_type'].str.lower() == 'ped']
ped_df = ped_df.dropna(subset=['long', 'lat'])

features = []
for _, row in ped_df.iterrows():
    lon = row['long']
    lat = row['lat']

    props = row.drop(['long', 'lat']).where(pd.notnull(row), None).to_dict()

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "properties": props
    }
    features.append(feature)

geojson_dict = {
    "type": "FeatureCollection",
    "features": features
}

output_path = 'model/datasets/crash/pedestrian_crashes.geojson'
with open(output_path, 'w') as f:
    json.dump(geojson_dict, f, indent=2)
