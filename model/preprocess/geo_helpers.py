import pandas as pd

# Create a boolean column if crash within a certain distance
def has_crash_near(point, crashes, radius):
    buffer = point.buffer(radius)
    return crashes.intersects(buffer).any()

# Reusable boolean proximity function
def count_nearby_features(point, features_gdf, radius):
    buffer = point.buffer(radius)
    possible_matches_index = list(features_gdf.sindex.intersection(buffer.bounds))
    possible_matches = features_gdf.iloc[possible_matches_index]
    return possible_matches.intersects(buffer).sum()

# Reusable non-binary proximity function that 
# if a feature is within 10 m of a point either give the average value 
# in the specified column name close to the point
# or give the number of features
def nearby_feature_value(point, features_gdf, radius, column_name=None):
    buffer = point.buffer(radius)
    possible_matches_index = list(features_gdf.sindex.intersection(buffer.bounds))
    candidates = features_gdf.iloc[possible_matches_index]
    nearby = candidates[candidates.intersects(buffer)]

    if nearby.empty:
        return 0 

    if column_name is None:
        return len(nearby)
    else:
        nearby = nearby.copy()
        nearby[column_name] = pd.to_numeric(
            nearby[column_name], errors='coerce'
        )
        return nearby[column_name].mean()
