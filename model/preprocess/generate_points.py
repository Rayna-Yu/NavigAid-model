import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString

G = ox.graph_from_place('Boston, Massachusetts, USA', network_type='walk')
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
edges = edges.drop_duplicates(subset='geometry')
edges = edges.to_crs(epsg=32619)

def sample_points_along_line(line: LineString, distance: float):
    points = []
    length = line.length
    if length < distance:
        return [line.interpolate(0), line.interpolate(length)]
    
    current_distance = 0.0
    while current_distance < length:
        points.append(line.interpolate(current_distance))
        current_distance += distance
    
    points.append(line.interpolate(length))
    return points

# Sample points every 1000 meters
sampled_points = []
total_points = 0

for idx, row in edges.iterrows():
    line = row['geometry']
    points = sample_points_along_line(line, 1000)
    total_points += len(points)
    sampled_points.extend(points)

gdf_points = gpd.GeoDataFrame(geometry=sampled_points, crs='EPSG:32619')
gdf_points = gdf_points.to_crs(epsg=4326)

# Randomly sample 300,000 points (without replacement)
if len(gdf_points) > 300000:
    gdf_points = gdf_points.sample(n=300000, random_state=42).reset_index(drop=True)

# Save
gdf_points.to_file("model/datasets/sample/boston_sampled_points.geojson", driver='GeoJSON')
gdf_points['lon'] = gdf_points.geometry.x
gdf_points['lat'] = gdf_points.geometry.y
gdf_points[['lat', 'lon']].to_csv("model/datasets/sample/boston_sampled_points.csv", index=False)