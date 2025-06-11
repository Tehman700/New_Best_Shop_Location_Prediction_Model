from matplotlib import pyplot as plt
from data_collection import combined_gdf, food_shops, pois
import geopandas as gpd
import os

def create_grid(combined_gdf, cell_size=0.005):
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
