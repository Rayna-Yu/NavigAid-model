import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString

# constants
crash_buffer_meters = 50
features_buffer_meters = 10

# helper functions
def normalize_value(val):
    no_spaces = val.replace(" ", "")
    if val is None:
        return None
    elif "/" in no_spaces:
        fraction = no_spaces.split("/")
        num = float(fraction[0])
        den = float(fraction[1])
        return num / den
    else:
        try:
            return float(no_spaces)
        except ValueError:
            return None

# import the boston sample points
points_gdf = gpd.read_file('model/datasets/sample/boston_sampled_points.geojson')

# import crash dataset
crashes_gdf = gpd.read_file('model/datasets/crash/pedestrian_crashes.geojson')

# import the feature datasets
segments_gdf = gpd.read_file('model/datasets/features/boston_segments.geojson')
ramps_gdf = gpd.read_file('model/datasets/features/pedestrian_ramp.geojson')
centerline_gdf = gpd.read_file('model/datasets/features/sidewalk_centerline.geojson')
sidewalks_gdf = gpd.read_file('model/datasets/features/sidewalks.geojson')
lamps_gdf = gpd.read_file('model/datasets/features/streetlight_locations.geojson')
trees_gdf = gpd.read_file('model/datasets/features/trees_data.geojson')

# Convert CRS to metric (meters) for distance calculations
points_gdf = points_gdf.to_crs(epsg=32619)
crashes_gdf = crashes_gdf.to_crs(epsg=32619)

# Create a boolean column if crash within 50m
def has_crash_near(point, crashes, radius):
    buffer = point.buffer(radius)
    return crashes.intersects(buffer).any()

points_gdf['label'] = points_gdf.geometry.apply(lambda pt: has_crash_near(pt, crashes_gdf.geometry, crash_buffer_meters))
points_gdf['label'] = points_gdf['label'].astype(int)

# Create a boolean column if there is a road with over 25mph within 15m
segments_gdf = segments_gdf.to_crs(epsg=32619)
high_speed_segments = segments_gdf[segments_gdf['SPEEDLIMIT'] > 25]

def has_high_speed_near(point, radius):
    buffer = point.buffer(radius)
    return high_speed_segments.intersects(buffer).any()

points_gdf['high_speed_limit'] = points_gdf.geometry.apply(
    lambda pt: int(has_high_speed_near(pt, features_buffer_meters))
)

# Ensure ramps, trees, and lamps are in the correct CRS
ramps_gdf = ramps_gdf.to_crs(epsg=32619)
trees_gdf = trees_gdf.to_crs(epsg=32619)
lamps_gdf = lamps_gdf.to_crs(epsg=32619)
centerline_gdf = centerline_gdf.to_crs(epsg=32619)

# Define reusable proximity function
def count_nearby_features(point, features_gdf, radius):
    buffer = point.buffer(radius)
    possible_matches_index = list(features_gdf.sindex.intersection(buffer.bounds))
    possible_matches = features_gdf.iloc[possible_matches_index]
    return possible_matches.intersects(buffer).sum()

# Flag points with at least one ramp nearby
points_gdf['has_ramp'] = points_gdf.geometry.apply(
    lambda pt: 1 if count_nearby_features(pt, ramps_gdf, features_buffer_meters) > 0 else 0
)

# Flag points with at least one tree nearby
points_gdf['has_tree'] = points_gdf.geometry.apply(
    lambda pt: 1 if count_nearby_features(pt, trees_gdf, features_buffer_meters) > 0 else 0
)

# Flag points with no lighting nearby
points_gdf['poor_lighting'] = points_gdf.geometry.apply(
    lambda pt: 1 if count_nearby_features(pt, lamps_gdf.geometry, features_buffer_meters) == 0 else 0
)

# Flag points with no sidewalk centerline nearby
points_gdf['no_sidewalk'] = points_gdf.geometry.apply(
    lambda pt: 1 if count_nearby_features(pt, centerline_gdf.geometry, features_buffer_meters) == 0 else 0
)


# Flag for sidewalk conditions
sidewalks_gdf = sidewalks_gdf.to_crs(epsg=32619)

assert points_gdf.crs == sidewalks_gdf.crs, "CRS mismatch between points and sidewalks"

def sidewalk_flag_near(point, sidewalks_gdf, radius, condition):
    buffer = point.buffer(radius)
    possible_matches = sidewalks_gdf.iloc[
        list(sidewalks_gdf.sindex.intersection(buffer.bounds))
    ]
    possible_matches = possible_matches[possible_matches.intersects(buffer)]

    for _, swk in possible_matches.iterrows():
        geom = swk.geometry

        if geom is None:
            continue

        # Must be near
        near = False
        if isinstance(geom, (Polygon, MultiPolygon, LineString, MultiLineString)):
            if geom.distance(point) <= radius:
                near = True

        if not near:
            continue

        # Apply specific condition
        if condition == "width":
            width = normalize_value(swk.get('SWK_WIDTH'))
            if width is not None and 0 < width < 5:
                return 1

        elif condition == "slope":
            slope = normalize_value(swk.get('SWK_SLOPE'))
            if slope is not None and slope > 5:
                return 1

        elif condition == "damage":
            dam_area = normalize_value(swk.get('DAM_AREA'))
            swk_area = normalize_value(swk.get('SWK_AREA'))
            if dam_area is not None and swk_area is not None:
                try:
                    if (dam_area / swk_area) > 0.25:
                        return 1
                except ZeroDivisionError:
                    pass

    return 0


# Now apply for each condition
points_gdf['narrow_sidewalk'] = points_gdf.geometry.apply(
    lambda pt: sidewalk_flag_near(pt, sidewalks_gdf, features_buffer_meters, "width")
)

points_gdf['steep_slope'] = points_gdf.geometry.apply(
    lambda pt: sidewalk_flag_near(pt, sidewalks_gdf, features_buffer_meters, "slope")
)

points_gdf['poor_condition'] = points_gdf.geometry.apply(
    lambda pt: sidewalk_flag_near(pt, sidewalks_gdf, features_buffer_meters, "damage")
)

# delete any row with None
cols_to_check = [
    'label', 'high_speed_limit', 'has_ramp', 'has_tree',
    'poor_lighting', 'no_sidewalk',
    'narrow_sidewalk', 'steep_slope', 'poor_condition'
]
points_gdf = points_gdf.dropna(subset=cols_to_check)

# Save to CSV (drop geometry)
points_gdf = points_gdf.to_crs(epsg=4326)
points_gdf.drop(columns='geometry').to_csv('model/datasets/final_csv/flags_and_crash_data.csv', index=False)

