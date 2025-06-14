# Islamabad & Taxila Grid Suitability Scoring Script
from matplotlib import pyplot as plt
from data_collection import combined_gdf, food_shops, pois
import geopandas as gpd
import os
import numpy as np
from shapely.geometry import box
import rasterio
from rasterstats import zonal_stats

def create_grid(combined_gdf, cell_size=0.01):
    bounds = combined_gdf.total_bounds
    xmin, ymin, xmax, ymax = bounds
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




# --------------------------
# Step 2: Suitability Score
# --------------------------
grid_cells["suitability_score"] = grid_cells["num_pois"] - grid_cells["num_food_shops"]

# --------------------------
# Step 3: Visualize Density & Score
# --------------------------
def plot_heatmap(column, title, cmap):
    fig, ax = plt.subplots(figsize=(12, 10))
    grid_cells.plot(ax=ax, column=column, cmap=cmap, legend=True)
    plt.title(title)
    plt.show()

plot_heatmap("num_food_shops", "Food Shop Density per Grid Cell", "Reds")
plot_heatmap("num_pois", "POI Density per Grid Cell", "Blues")
plot_heatmap("suitability_score", "Suitability Score for New Food Shop Locations", "viridis")

# --------------------------
# Step 4: Save Scored Grid
# --------------------------
grid_cells.to_file("data/processed/scored_grid.geojson", driver="GeoJSON")
print("✅ Scored grid saved at: data/processed/scored_grid.geojson")

# --------------------------
# Step 5: Add Population Density (Optional if raster is available)
# --------------------------
raster_path = "D:/New Shop Prediction Model/pakistan_raster.tif"

if os.path.exists(raster_path):
    print("📊 Calculating population density...")
    pop_stats = zonal_stats(
        grid_cells,
        raster_path,
        stats=["mean"],
        geojson_out=False,
        nodata=-999
    )
    grid_cells["pop_density"] = [stat["mean"] if stat else 0 for stat in pop_stats]

    # Plot population density
    plot_heatmap("pop_density", "Average Population Density per Grid Cell", "Oranges")

    # Optional: Recalculate suitability to include population weight
    grid_cells["weighted_score"] = grid_cells["suitability_score"] * grid_cells["pop_density"]
    plot_heatmap("weighted_score", "Weighted Suitability Score (incl. Pop. Density)", "plasma")

    # Save updated file
    grid_cells.to_file("data/processed/final_scored_grid.geojson", driver="GeoJSON")
    print("✅ Final scored grid with population saved at: data/processed/final_scored_grid.geojson")
else:
    print("⚠️ Population raster not found. Skipping population-based analysis.")








# Sort by highest weighted score
top_cells = grid_cells.sort_values(by="weighted_score", ascending=False)

# Filter: remove cells with NaN or zero score
top_cells = top_cells[top_cells["weighted_score"].notnull() & (top_cells["weighted_score"] > 0)]

# Select top 10
top_10 = top_cells.head(10)

# Save them to file
top_10.to_file("data/processed/top_10_locations.geojson", driver="GeoJSON")
print("📍 Top 10 locations saved to: data/processed/top_10_locations.geojson")


fig, ax = plt.subplots(figsize=(12, 10))
grid_cells.boundary.plot(ax=ax, color='lightgray', linewidth=0.5)
top_10.plot(ax=ax, color='green', edgecolor='black')
plt.title("Top 10 Recommended Locations for New Food Shops")
plt.show()


top_10[["num_food_shops", "num_pois", "suitability_score", "pop_density", "weighted_score"]].to_csv("data/processed/top_10_locations.csv", index=False)





from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = grid_cells[["num_food_shops", "num_pois", "pop_density"]].fillna(0)
y = grid_cells["weighted_score"].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

print("R² Score:", model.score(X_test, y_test))
