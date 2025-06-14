from matplotlib import pyplot as plt
from data_collection import combined_gdf, food_shops, pois
import geopandas as gpd
import os

def create_grid(combined_gdf, cell_size=0.01):
    bounds = combined_gdf.total_bounds
    xmin, ymin, xmax, ymax = bounds

    from shapely.geometry import box
    import numpy as np

    cols = list(np.arange(xmin, xmax, cell_size))
    rows = list(np.arange(ymin, ymax, cell_size))

    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(box(x, y, x + cell_size, y + cell_size))

    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=combined_gdf.crs)
    grid_clipped = gpd.clip(grid, combined_gdf)
    return grid_clipped

# Create the grid cells
grid_cells = create_grid(combined_gdf)

# Ensure output directory exists before saving
os.makedirs("data/processed", exist_ok=True)
grid_cells.to_file("data/processed/grid_cells.geojson", driver="GeoJSON")

# Plotting
fig, ax = plt.subplots(figsize=(12, 10))
combined_gdf.plot(ax=ax, color='white', edgecolor='black')
grid_cells.boundary.plot(ax=ax, color='grey', alpha=0.5)
food_shops.plot(ax=ax, color='red', markersize=5)
pois.plot(ax=ax, color='blue', markersize=5, label='POIs')
plt.title("Grid over Islamabad & Taxila")
plt.show()


# --- Step 1: Count Food Shops and POIs per Grid Cell ---

# Count food shops in each grid cell
food_join = gpd.sjoin(grid_cells, food_shops, predicate="contains", how="left")
food_count = food_join.groupby(food_join.index).size()
grid_cells["num_food_shops"] = grid_cells.index.map(food_count).fillna(0)

# Count POIs in each grid cell
poi_join = gpd.sjoin(grid_cells, pois, predicate="contains", how="left")
poi_count = poi_join.groupby(poi_join.index).size()
grid_cells["num_pois"] = grid_cells.index.map(poi_count).fillna(0)


# --- Step 2: Calculate Suitability Score ---
# You can customize the scoring formula here
grid_cells["suitability_score"] = grid_cells["num_pois"] - grid_cells["num_food_shops"]

# --- Step 3: Visualize Heatmaps ---

# Food shop density heatmap
fig, ax = plt.subplots(figsize=(12, 10))
grid_cells.plot(ax=ax, column="num_food_shops", cmap="Reds", legend=True)
plt.title("Food Shop Density per Grid Cell")
plt.show()

# POI density heatmap
fig, ax = plt.subplots(figsize=(12, 10))
grid_cells.plot(ax=ax, column="num_pois", cmap="Blues", legend=True)
plt.title("POI Density per Grid Cell")
plt.show()

# Final suitability score heatmap
fig, ax = plt.subplots(figsize=(12, 10))
grid_cells.plot(ax=ax, column="suitability_score", cmap="viridis", legend=True)
plt.title("Suitability Score for New Food Shop Locations")
plt.show()

# --- Step 4: Save the Scored Grid ---

grid_cells.to_file("data/processed/scored_grid.geojson", driver="GeoJSON")
print("✅ Scored grid saved at: data/processed/scored_grid.geojson")




