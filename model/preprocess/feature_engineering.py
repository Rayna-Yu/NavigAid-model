import geopandas as gpd
import pandas as pd
from geo_helpers import has_crash_near, count_nearby_features, nearby_feature_value
from utils import normalize_value

def add_feature_flags(points_gdf, crashes_gdf, datasets, buffer_m, is_continuous):
    # Label crash proximity
    points_gdf['label'] = points_gdf.geometry.apply(lambda pt: has_crash_near(pt, crashes_gdf.geometry, buffer_m)).astype(int)

    # handle the feature variables based off of whether we want continuous data or boolean data
    if (is_continuous):
       return _balance_classes(_handle_continuous(points_gdf, datasets, buffer_m))
    else:
       return _balance_classes(_handle_boolean(points_gdf, datasets, buffer_m))
    
# handles continuous data
def _handle_continuous(points_gdf, datasets, buffer_m):
    # speed limit
    points_gdf['speed_limit'] = points_gdf.geometry.apply(
        lambda pt: nearby_feature_value(pt, datasets['segments'], buffer_m, "SPEEDLIMIT")
    )

    # Ramp
    points_gdf['ramp'] = points_gdf.geometry.apply(
        lambda pt: nearby_feature_value(pt, datasets['ramps'], buffer_m)
    )

    # Trees
    points_gdf['tree'] = points_gdf.geometry.apply(
        lambda pt: nearby_feature_value(pt, datasets['trees'], buffer_m)
    )

    # Lighting
    points_gdf['lighting'] = points_gdf.geometry.apply(
        lambda pt: nearby_feature_value(pt,  datasets['lamps'], buffer_m)
    )

    # No sidewalk
    points_gdf['has_crosswalk'] = points_gdf.geometry.apply(
        lambda pt: nearby_feature_value(pt, datasets['crosswalk'], buffer_m)
    )

    # Sidewalk conditions
    points_gdf['narrow_sidewalk'] = points_gdf.geometry.apply(
        lambda pt: nearby_feature_value(pt, datasets['sidewalks'], buffer_m, "SWK_WIDTH")
    )

    points_gdf['steep_slope'] = points_gdf.geometry.apply(
        lambda pt: nearby_feature_value(pt, datasets['sidewalks'], buffer_m, "SWK_SLOPE")
    )

    sidewalks_gdf = datasets['sidewalks'].copy()
    sidewalks_gdf['DAM_AREA'] = sidewalks_gdf['DAM_AREA'].apply(normalize_value)
    sidewalks_gdf['SWK_AREA'] = sidewalks_gdf['SWK_AREA'].apply(normalize_value)
    sidewalks_gdf['DAM_AREA'] = pd.to_numeric(sidewalks_gdf['DAM_AREA'], errors='coerce')
    sidewalks_gdf['SWK_AREA'] = pd.to_numeric(sidewalks_gdf['SWK_AREA'], errors='coerce')
    sidewalks_gdf['DMG_PROP'] = sidewalks_gdf['DAM_AREA'] / sidewalks_gdf['SWK_AREA']
    sidewalks_gdf['DMG_PROP'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    points_gdf['poor_condition'] = points_gdf.geometry.apply(
        lambda pt: nearby_feature_value(pt, sidewalks_gdf, buffer_m, "DMG_PROP")
    )

    return points_gdf.dropna()

# handles boolean data
def _handle_boolean(points_gdf, datasets, buffer_m):
    # speed limit
    segments_gdf = datasets['segments']
    high_speed_segments = segments_gdf[segments_gdf['SPEEDLIMIT'] > 25]
    points_gdf['high_speed_limit'] = points_gdf.geometry.apply(
        lambda pt: int(count_nearby_features(pt, high_speed_segments, buffer_m))
    )

    # Ramp
    points_gdf['has_ramp'] = points_gdf.geometry.apply(
        lambda pt: 1 if count_nearby_features(pt, datasets['ramps'], buffer_m) > 0 else 0
    )

    # Trees
    points_gdf['has_tree'] = points_gdf.geometry.apply(
        lambda pt: 1 if count_nearby_features(pt, datasets['trees'], buffer_m) > 0 else 0
    )

    # Lighting
    points_gdf['poor_lighting'] = points_gdf.geometry.apply(
        lambda pt: 1 if count_nearby_features(pt, datasets['lamps'], buffer_m) == 0 else 0
    )

    # has crosswalk nearby
    points_gdf['has_crosswalk'] = points_gdf.geometry.apply(
        lambda pt: 1 if count_nearby_features(pt, datasets['crosswalk'], buffer_m) > 0 else 0
    )

    # Sidewalk conditions
    sidewalks_gdf = datasets['sidewalks'].copy()
    sidewalks_gdf['SWK_WIDTH'] = sidewalks_gdf['SWK_WIDTH'].apply(normalize_value)
    sidewalks_gdf['SWK_SLOPE'] = sidewalks_gdf['SWK_SLOPE'].apply(normalize_value)
    sidewalks_gdf['DAM_AREA'] = sidewalks_gdf['DAM_AREA'].apply(normalize_value)
    sidewalks_gdf['SWK_AREA'] = sidewalks_gdf['SWK_AREA'].apply(normalize_value)

    narrow_sidewalks = sidewalks_gdf[
        (sidewalks_gdf['SWK_WIDTH']).notnull() & 
        (sidewalks_gdf['SWK_WIDTH'] > 0) & 
        (sidewalks_gdf['SWK_WIDTH'] < 5)
    ]
    points_gdf['narrow_sidewalk'] = points_gdf.geometry.apply(
        lambda pt: 1 if count_nearby_features(pt, narrow_sidewalks, buffer_m) > 0 else 0
    )

    steep_sidewalks = sidewalks_gdf[
        (sidewalks_gdf['SWK_SLOPE']).notnull() & 
        (sidewalks_gdf['SWK_SLOPE'] > 5)
    ]
    points_gdf['steep_slope'] = points_gdf.geometry.apply(
        lambda pt: 1 if count_nearby_features(pt, steep_sidewalks, buffer_m) > 0 else 0
    )

    damage_sidewalks = sidewalks_gdf[
        (sidewalks_gdf['DAM_AREA']).notnull() & 
        (sidewalks_gdf['SWK_AREA']).notnull() & 
        (sidewalks_gdf['DAM_AREA']).notnull() & 
        (sidewalks_gdf['SWK_AREA'] != 0) & 
        (sidewalks_gdf['DAM_AREA'] / sidewalks_gdf['SWK_AREA']) > 0.25
    ]
    points_gdf['damaged_sidewalk'] = points_gdf.geometry.apply(
        lambda pt: 1 if count_nearby_features(pt, damage_sidewalks, buffer_m) > 0 else 0
    )

    return points_gdf.dropna()

def _balance_classes(points_gdf):
    crash_points = points_gdf[points_gdf['label'] == 1]
    non_crash_points = points_gdf[points_gdf['label'] == 0]
    min_size = min(len(crash_points), len(non_crash_points))

    sampled_crashes = crash_points.sample(n=min_size, random_state=42)
    sampled_non_crashes = non_crash_points.sample(n=min_size, random_state=42)

    balanced_points = pd.concat([sampled_crashes, sampled_non_crashes], ignore_index=True)
    return balanced_points.sample(frac=1, random_state=42).reset_index(drop=True)
