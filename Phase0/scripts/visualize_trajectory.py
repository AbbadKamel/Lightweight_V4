"""
Visualize boat trajectory from decoded NMEA2000 data
Creates an interactive HTML map showing the boat's path

HOW TO RUN:
    cd /path/to/Lightweight_IA_V_3
    python3 Phase0/scripts/visualize_trajectory.py

OUTPUT:
    - Phase0/results/boat_trajectory.html

TO VIEW MAP:
    python3 -m http.server 8888
    Then open: http://localhost:8888/Phase0/results/boat_trajectory.html
"""
import pandas as pd
import json

# Read decoded data
print("Loading decoded data...")
df = pd.read_csv('Phase0/results/decoded_frames.csv')

# Filter only position data
positions = df[df['pgn'] == 129025][['timestamp', 'latitude', 'longitude', 'pgn_name']].dropna()

# Sample every Nth point to keep file small (plot every 10th point)
positions = positions[::10].reset_index(drop=True)

print(f"Found {len(positions):,} position points (sampled from {len(df[df['pgn'] == 129025]):,})")
print(f"Latitude range: {positions['latitude'].min():.6f} to {positions['latitude'].max():.6f}")
print(f"Longitude range: {positions['longitude'].min():.6f} to {positions['longitude'].max():.6f}")

# Convert to list for JavaScript
coords = positions[['latitude', 'longitude', 'timestamp']].values.tolist()

# Create HTML map using Leaflet.js (no API key needed)
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>NMEA2000 Boat Trajectory Visualization</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #map {{
            position: absolute;
            top: 0;
            bottom: 60px;
            width: 100%;
        }}
        #info {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: rgba(255, 255, 255, 0.95);
            padding: 10px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            align-items: center;
            height: 100%;
        }}
        .stat-box {{
            text-align: center;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
        }}
        .stat-value {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .legend {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="info">
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Total Points</div>
                <div class="stat-value">{len(positions):,}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Latitude Range</div>
                <div class="stat-value">{positions['latitude'].min():.4f}° - {positions['latitude'].max():.4f}°</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Longitude Range</div>
                <div class="stat-value">{positions['longitude'].min():.4f}° - {positions['longitude'].max():.4f}°</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Data Source</div>
                <div class="stat-value">NMEA2000 PGN 129025</div>
            </div>
        </div>
    </div>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <script>
        // Boat trajectory coordinates
        var coordinates = {json.dumps(coords)};
        
        // Calculate center
        var lats = coordinates.map(c => c[0]);
        var lons = coordinates.map(c => c[1]);
        var centerLat = (Math.min(...lats) + Math.max(...lats)) / 2;
        var centerLon = (Math.min(...lons) + Math.max(...lons)) / 2;
        
        // Initialize map
        var map = L.map('map').setView([centerLat, centerLon], 13);
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }}).addTo(map);
        
        // Create polyline for trajectory
        var latlngs = coordinates.map(c => [c[0], c[1]]);
        var polyline = L.polyline(latlngs, {{
            color: '#2563eb',
            weight: 3,
            opacity: 0.7
        }}).addTo(map);
        
        // Start marker (green)
        L.circleMarker([coordinates[0][0], coordinates[0][1]], {{
            radius: 8,
            fillColor: '#10b981',
            color: '#ffffff',
            weight: 2,
            fillOpacity: 1
        }}).addTo(map)
          .bindPopup('<b>Start Position</b><br>Time: ' + coordinates[0][2]);
        
        // End marker (red)
        var lastIdx = coordinates.length - 1;
        L.circleMarker([coordinates[lastIdx][0], coordinates[lastIdx][1]], {{
            radius: 8,
            fillColor: '#ef4444',
            color: '#ffffff',
            weight: 2,
            fillOpacity: 1
        }}).addTo(map)
          .bindPopup('<b>End Position</b><br>Time: ' + coordinates[lastIdx][2]);
        
        // Add every 100th point as small marker
        for (var i = 0; i < coordinates.length; i += 100) {{
            L.circleMarker([coordinates[i][0], coordinates[i][1]], {{
                radius: 3,
                fillColor: '#3b82f6',
                color: '#ffffff',
                weight: 1,
                fillOpacity: 0.8
            }}).addTo(map)
              .bindPopup('Time: ' + coordinates[i][2]);
        }}
        
        // Fit map to trajectory
        map.fitBounds(polyline.getBounds(), {{padding: [50, 50]}});
        
        // Add legend
        var legend = L.control({{position: 'topright'}});
        legend.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<h4 style="margin:0 0 5px 0">Boat Trajectory</h4>' +
                          '<div><span style="color:#10b981">●</span> Start</div>' +
                          '<div><span style="color:#2563eb">—</span> Path</div>' +
                          '<div><span style="color:#ef4444">●</span> End</div>';
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>"""

# Save HTML file
output_file = 'Phase0/results/boat_trajectory.html'
with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n✓ Created: {output_file}")
print(f"\nTo view the map:")
print(f"  1. Open in browser: file://{output_file}")
print(f"  2. Or run: python3 -m http.server 8000")
print(f"     Then visit: http://localhost:8000/Phase0/results/boat_trajectory.html")
