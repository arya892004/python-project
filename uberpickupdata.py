import random
import pandas as pd
from datetime import datetime
import folium
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from folium.plugins import HeatMap

# Step 1: Generate Synthetic Uber Pickup Data for Delhi

# Function to generate synthetic data
def generate_synthetic_data(num_records):
    locations = [
        (28.6448, 77.216721), (28.6445, 77.2100), (28.6480, 77.2295), 
        (28.6270, 77.2190), (28.6368, 77.2178), (28.6420, 77.2333)
    ]  # Sample lat, lon pairs (Delhi area)

    data = []
    for _ in range(num_records):
        pickup_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location = random.choice(locations)
        data.append([pickup_time, location[0], location[1]])

    return pd.DataFrame(data, columns=['Pickup Time', 'Latitude', 'Longitude'])

# Generate 1000 synthetic records
df = generate_synthetic_data(1000)

# Step 2: Data Preprocessing
df['Pickup Time'] = pd.to_datetime(df['Pickup Time'])  # Convert Pickup Time to datetime

# Step 3: Hotspot Analysis (K-Means Clustering)

# Use KMeans to find pickup hotspots
kmeans = KMeans(n_clusters=5)  # Choose 5 clusters as the number of hotspots
df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])

# Show the cluster centers (hotspot locations)
print("Cluster Centers (Hotspots):")
print(kmeans.cluster_centers_)

# Step 4: Hourly Pickup Patterns

df['Hour'] = df['Pickup Time'].dt.hour  # Extract hour from the Pickup Time
hourly_pickups = df.groupby('Hour').size()

# Step 5: Data Visualization

# 1. Heatmap of Uber Pickups using Folium
map_deli = folium.Map(location=[28.6448, 77.216721], zoom_start=12)

# Add HeatMap
heat_data = [[row['Latitude'], row['Longitude']] for _, row in df.iterrows()]
HeatMap(heat_data).add_to(map_deli)

# Save the heatmap to an HTML file
map_deli.save("delhi_uber_heatmap.html")
print("Heatmap saved as delhi_uber_heatmap.html")

# 2. Map of Uber Pickup Locations with Clusters
map_deli_cluster = folium.Map(location=[28.6448, 77.216721], zoom_start=12)

# Add pickup locations and cluster centers to the map
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']], 
        radius=3, color='blue', fill=True
    ).add_to(map_deli_cluster)

# Mark the hotspot (cluster) centers
for center in kmeans.cluster_centers_:
    folium.Marker(location=[center[0], center[1]], popup="Hotspot", icon=folium.Icon(color='red')).add_to(map_deli_cluster)

# Save the cluster map to an HTML file
map_deli_cluster.save("delhi_uber_clusters.html")
print("Cluster Map saved as delhi_uber_clusters.html")

# 3. Hourly Pickup Distribution Bar Chart
hourly_pickups.plot(kind='bar', color='skyblue')
plt.title('Hourly Pickup Distribution in Delhi')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Pickups')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the bar chart as an image
plt.savefig('hourly_pickups.png')
plt.show()
print("Hourly Pickup Distribution saved as hourly_pickups.png")
