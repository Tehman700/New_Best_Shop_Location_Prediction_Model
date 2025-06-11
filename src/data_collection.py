import shutil
import os

folder_path = r"D:\New Shop Prediction Model\src\data"
t = r"D:\New Shop Prediction Model\src\cache"# Use raw string or double backslashes on Windows

# Check if the folder exists
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
    print(f"Deleted folder Data: {folder_path}")
else:
    print("Folder does not exist.")
if os.path.exists(t):
    shutil.rmtree(t)
    print(f"Deleted folder Cache : {t}")
else:
    print("Folder does not exist.")

if os.path.exists(folder_path):
    print("NOT DELETED")
elif os.path.exists(t):
    print("NOT DELETED")

import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Settings
ox.settings.log_console = True
ox.settings.use_cache = True

# -------------------------
# 1. Define the Area
# -------------------------
place_name = "Islamabad, Pakistan"
taxila_name = "Taxila Tehsil, Punjab, Pakistan"

# You can combine both regions into one
islamabad = ox.geocode_to_gdf(place_name)
taxila = ox.geocode_to_gdf(taxila_name)

# Combine into one bounding box
combined_area = gpd.GeoDataFrame(pd.concat([islamabad, taxila], ignore_index=True))
bounding_polygon = combined_area.geometry.union_all()

# -------------------------
# 2. Collect Food Shops
# -------------------------
food_tags = {
    "amenity": ["restaurant", "fast_food", "cafe", "hotels", "cuisines"]
}

food_shops = ox.features_from_polygon(bounding_polygon, tags=food_tags)

# -------------------------
# 3. Collect Other POIs (optional but useful)
# -------------------------
poi_tags = {
    "amenity": ["school", "college", "bus_station", "university", "marketplace"],
    "shop": ["supermarket", "mall"]
}

pois = ox.features_from_polygon(bounding_polygon, tags=poi_tags)

# -------------------------
# 4. Save the Data
# -------------------------
os.makedirs("data/raw", exist_ok=True)

food_shops.to_file("data/raw/food_shops.geojson", driver='GeoJSON')
pois.to_file("data/raw/pois.geojson", driver='GeoJSON')

# -------------------------
# 5. Plot for Quick Check
# -------------------------
fig, ax = plt.subplots(figsize=(12, 10))
islamabad.plot(ax=ax, color='lightgrey')
taxila.plot(ax=ax, color='lightgrey')
food_shops.plot(ax=ax, color='red', markersize=5, label="Food Shops")
pois.plot(ax=ax, color='blue', markersize=5, label="POIs")
plt.title("Food Shops and POIs in Islamabad & Taxila")
plt.show()


combined_gdf = gpd.GeoDataFrame(geometry=[bounding_polygon], crs="EPSG:4326")

# Now below part is going to create a grid for this polygon in the other file
# Creating_grid.py

