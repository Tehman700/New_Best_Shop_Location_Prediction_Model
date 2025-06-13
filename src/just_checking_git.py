import pandas as pd

# Load datasets
df_categories = pd.read_csv("D:/ML Dataset/category_tree.csv")
df_events = pd.read_csv("D:/ML Dataset/events.csv")
df_props1 = pd.read_csv("D:/ML Dataset/item_properties_part1.csv")
df_props2 = pd.read_csv("D:/ML Dataset/item_properties_part2.csv")




print(df_props1.shape)
print(df_props2.describe())
# Combine item properties
# df_item_props = pd.concat([df_props1, df_props2])
#
# # Filter only categoryid properties and rename
# df_cat_props = df_item_props[df_item_props['property'] == 'categoryid'][['itemid', 'value']].drop_duplicates()
# df_cat_props.rename(columns={'value': 'categoryid'}, inplace=True)
#
# # Merge categoryid into events
# df_events = df_events.merge(df_cat_props, on='itemid', how='left')
#
# # Convert timestamp to datetime
# df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='ms')
#
# # Feature Engineering per user
# user_stats = df_events.groupby('visitorid').agg(
#     total_views=('event', lambda x: (x == 'view').sum()),
#     total_adds=('event', lambda x: (x == 'addtocart').sum()),
#     total_purchases=('event', lambda x: (x == 'transaction').sum()),
#     unique_items=('itemid', pd.Series.nunique),
#     unique_categories=('categoryid', pd.Series.nunique),
#     first_time=('timestamp', 'min'),
#     last_time=('timestamp', 'max'),
#     total_events=('event', 'count')
# ).reset_index()
#
# # Time-based features
# user_stats['active_seconds'] = (user_stats['last_time'] - user_stats['first_time']).dt.total_seconds()
# user_stats['avg_seconds_per_event'] = user_stats['active_seconds'] / user_stats['total_events'].replace(0, 1)
#
# # Ratios
# user_stats['view_to_cart'] = user_stats['total_views'] / user_stats['total_adds'].replace(0, 1)
# user_stats['cart_to_purchase'] = user_stats['total_adds'] / user_stats['total_purchases'].replace(0, 1)
#
# # Rule-based abnormal behavior labeling
# user_stats['abnormal'] = 0
# user_stats.loc[(user_stats['total_views'] > 50) & (user_stats['total_purchases'] == 0), 'abnormal'] = 1
# user_stats.loc[(user_stats['avg_seconds_per_event'] < 2), 'abnormal'] = 1  # possible bot-like behavior
# user_stats.loc[(user_stats['view_to_cart'] > 20), 'abnormal'] = 1  # many views, almost no interest
# user_stats.loc[(user_stats['unique_categories'] > 50), 'abnormal'] = 1  # unusual browsing patterns
#
# # Show sample
# print(user_stats[['visitorid', 'total_views', 'total_adds', 'total_purchases', 'unique_items', 'avg_seconds_per_event', 'abnormal']].head(10))
