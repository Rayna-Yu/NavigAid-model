import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString

# Get the street network graph for Boston (driveable or walkable)
G = ox.graph_from_place('Boston, Massachusetts, USA', network_type='walk')

# Convert the graph edges to a GeoDataFrame
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# Function to sample points evenly spaced along a LineString
def sample_points_along_line(line: LineString, distance: float):
    points = []
    length = line.length
    current_distance = 0.0
    while current_distance < length:
        point = line.interpolate(current_distance)
        points.append(point)
        current_distance += distance
    # add last point
    points.append(line.interpolate(length))
    return points

# Sample points every 20 meters on each edge
sampled_points = []
for idx, row in edges.iterrows():
    line = row['geometry']
    points = sample_points_along_line(line, 100)
    sampled_points.extend(points)

# Create GeoDataFrame of sampled points
gdf_points = gpd.GeoDataFrame(geometry=sampled_points, crs='EPSG:4326')

print(f"Total points sampled: {len(gdf_points)}")

# Optionally save to GeoJSON or CSV for your next steps
gdf_points.to_file("boston_sampled_points.geojson", driver='GeoJSON')
gdf_points['lon'] = gdf_points.geometry.x
gdf_points['lat'] = gdf_points.geometry.y
gdf_points[['lat', 'lon']].to_csv("boston_sampled_points.csv", index=False)
