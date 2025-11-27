import pandas as pd
import json
import os

# Ensure output directory exists
os.makedirs('Phase0/results', exist_ok=True)

print("Loading data...")
# df = pd.read_csv('Phase0/results/decoded_frames_aligned.csv')
df = pd.read_csv('Phase0/results/decoded_frames_realtime.csv')
# df_pos = pd.read_csv('Phase0/results/decoded_frames.csv')

# Filter positions from RAW data (cleaner)
# raw_pos = df_pos[df_pos['pgn'] == 129025][['latitude', 'longitude', 'timestamp']].dropna().reset_index(drop=True)


# Parse timestamps
def parse_time(ts):
    try:
        parts = str(ts).split(':')
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2]) # No need to replace .0 anymore, new CSV is clean
        return h * 3600 + m * 60 + s
    except:
        return 0

# Calculate relative time for raw positions
# start_time = parse_time(raw_pos['timestamp'].iloc[0])
# raw_pos['time_rel'] = raw_pos['timestamp'].apply(parse_time) - start_time

# Downsample raw positions to ~1Hz (approx every 10 points if 10Hz)
# Better: Group by rounded relative time to match aligned data
# raw_pos['time_rounded'] = raw_pos['time_rel'].round().astype(int)
# clean_pos = raw_pos.groupby('time_rounded').first().reset_index()

# print(f"Extracted {len(clean_pos)} clean position points from raw data")

# Limit aligned data to valid duration
# duration = raw_pos['time_rel'].max()
# df = df[df['time_rounded'] <= duration].reset_index(drop=True)

print(f"Aligned data has {len(df)} rows")

# Merge clean positions with aligned sensors
# We use the 'time_rounded' column to join
# merged = pd.merge(clean_pos, df, on='time_rounded', how='inner', suffixes=('_raw', '_aligned'))
merged = df # Direct use of the new realtime dataframe

# Downsample for visualization (otherwise 260k points will crash the browser)
# Take 1 point every 10 rows (approx 1-2Hz)
merged = merged.iloc[::10].reset_index(drop=True)

print(f"Merged data has {len(merged)} points")

data = []
for i in range(len(merged)):
    row = merged.iloc[i]
    
    # Use RAW latitude/longitude (clean)
    lat = row['latitude']
    lon = row['longitude']
    
    def get_val(row, col, default=0):
        val = row.get(col)
        return float(val) if pd.notna(val) else default

    data.append({
        'lat': lat,
        'lon': lon,
        'sog': get_val(row, 'sog'),
        'heading': get_val(row, 'heading'),
        'cog': get_val(row, 'cog'),
        'depth': get_val(row, 'depth'),
        'wind_speed': get_val(row, 'wind_speed'),
        'rudder': get_val(row, 'rudder_position'),
        'pitch': get_val(row, 'pitch'),
        'roll': get_val(row, 'roll')
    })

# Define fixed scales based on data analysis
scales = {
    'sog': {'min': 0, 'max': 3},
    'heading': {'min': 0, 'max': 360},
    'depth': {'min': 0, 'max': 15},
    'wind': {'min': 0, 'max': 10},
    'rudder': {'min': -70, 'max': 70},
    'pitchroll': {'min': -5, 'max': 5}
}

html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Boat Trajectory - Controlled View</title>
    <meta charset="utf-8" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; font-size: 12px; }}
        .container {{ 
            display: grid; 
            grid-template-columns: 1.5fr 1fr 1fr; 
            grid-template-rows: 60px 1fr 1fr 1fr; 
            gap: 8px; 
            height: 100vh; 
            padding: 8px; 
            box-sizing: border-box; 
        }}
        
        /* Controls Header */
        #controls {{ 
            grid-column: 1 / 4; 
            grid-row: 1; 
            background: white; 
            padding: 5px 15px; 
            border-radius: 6px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
            display: flex; 
            flex-direction: column; 
            justify-content: center;
            gap: 5px;
        }}
        
        .control-row {{
            display: flex;
            align-items: center;
            gap: 15px;
            width: 100%;
        }}
        
        #timeline {{
            width: 100%;
            cursor: pointer;
            margin: 0;
        }}
        
        /* Map Area */
        #map {{ 
            grid-column: 1; 
            grid-row: 2 / 5; 
            border-radius: 6px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
            background: #ddd;
        }}
        
        /* Charts */
        .chart-wrapper {{ 
            background: white; 
            padding: 8px; 
            border-radius: 6px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
            display: flex; 
            flex-direction: column; 
            position: relative;
            overflow: hidden;
        }}
        
        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
            border-bottom: 1px solid #eee;
            padding-bottom: 2px;
        }}
        
        .chart-title {{ font-weight: 600; color: #374151; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }}
        
        .toggle-scale {{ 
            font-size: 9px; 
            cursor: pointer; 
            background: #e5e7eb; 
            border: none; 
            padding: 2px 6px; 
            border-radius: 3px; 
            color: #4b5563;
        }}
        .toggle-scale:hover {{ background: #d1d5db; }}
        .toggle-scale.active {{ background: #3b82f6; color: white; }}

        canvas {{ flex: 1; min-height: 0; }}
        
        /* UI Elements */
        button {{ padding: 6px 12px; font-size: 13px; border: none; border-radius: 4px; cursor: pointer; font-weight: 500; transition: background 0.2s; }}
        #playBtn {{ background: #2563eb; color: white; }}
        #playBtn:hover {{ background: #1d4ed8; }}
        #resetBtn {{ background: #ef4444; color: white; }}
        #resetBtn:hover {{ background: #dc2626; }}
        
        .stat-box {{ background: #f3f4f6; padding: 4px 8px; border-radius: 4px; font-family: monospace; font-size: 12px; }}
        
        input[type=range] {{ cursor: pointer; }}
    </style>
</head>
<body>
    <div class="container">
        <div id="controls">
            <div class="control-row">
                <button id="playBtn" onclick="togglePlay()">▶ Play</button>
                <button id="resetBtn" onclick="reset()">⏮ Reset</button>
                
                <div style="display:flex; align-items:center; gap:5px; margin-left:10px;">
                    <label>Speed:</label>
                    <input type="range" min="1" max="100" value="50" oninput="setSpeed(this.value)" style="width: 100px;">
                    <span id="speedVal">1x</span>
                </div>
                
                <div style="flex:1"></div>
                
                <div class="stat-box">IDX: <span id="idxDisplay">0</span> / {len(data)}</div>
            </div>
            <input type="range" id="timeline" min="0" max="{len(data)-1}" value="0" oninput="seek(this.value)">
        </div>

        <div id="map"></div>

        <!-- Row 2 Charts -->
        <div class="chart-wrapper" style="grid-column: 2; grid-row: 2;">
            <div class="chart-header">
                <div class="chart-title">Speed (SOG)</div>
                <button class="toggle-scale active" onclick="toggleScale('sog', this)">Fixed</button>
            </div>
            <canvas id="chart_sog"></canvas>
        </div>
        
        <div class="chart-wrapper" style="grid-column: 3; grid-row: 2;">
            <div class="chart-header">
                <div class="chart-title">Heading / COG</div>
                <button class="toggle-scale active" onclick="toggleScale('heading', this)">Fixed</button>
            </div>
            <canvas id="chart_heading"></canvas>
        </div>

        <!-- Row 3 Charts -->
        <div class="chart-wrapper" style="grid-column: 2; grid-row: 3;">
            <div class="chart-header">
                <div class="chart-title">Depth</div>
                <button class="toggle-scale active" onclick="toggleScale('depth', this)">Fixed</button>
            </div>
            <canvas id="chart_depth"></canvas>
        </div>
        
        <div class="chart-wrapper" style="grid-column: 3; grid-row: 3;">
            <div class="chart-header">
                <div class="chart-title">Wind Speed</div>
                <button class="toggle-scale active" onclick="toggleScale('wind', this)">Fixed</button>
            </div>
            <canvas id="chart_wind"></canvas>
        </div>

        <!-- Row 4 Charts -->
        <div class="chart-wrapper" style="grid-column: 2; grid-row: 4;">
            <div class="chart-header">
                <div class="chart-title">Rudder</div>
                <button class="toggle-scale active" onclick="toggleScale('rudder', this)">Fixed</button>
            </div>
            <canvas id="chart_rudder"></canvas>
        </div>
        
        <div class="chart-wrapper" style="grid-column: 3; grid-row: 4;">
            <div class="chart-header">
                <div class="chart-title">Pitch / Roll</div>
                <button class="toggle-scale active" onclick="toggleScale('pitchroll', this)">Fixed</button>
            </div>
            <canvas id="chart_pitchroll"></canvas>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Data Injection
        const data = {json.dumps(data)};
        const scales = {json.dumps(scales)};
        
        // State
        let idx = 0;
        let playing = false;
        let timer = null;
        let speed = 50; // ms delay
        const historyLen = 1000;
        
        // Map Init
        const map = L.map('map').setView([data[0].lat, data[0].lon], 14);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(map);
        const boatIcon = L.divIcon({{html: '🚤', className: 'boat-marker', iconSize: [24,24], iconAnchor: [12,12]}});
        const marker = L.marker([data[0].lat, data[0].lon], {{icon: boatIcon}}).addTo(map);
        const trail = L.polyline([], {{color: '#2563eb', weight: 2}}).addTo(map);
        
        // Chart Init
        const commonOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {{ legend: {{ display: true, labels: {{ boxWidth: 8, font: {{ size: 10 }} }} }} }},
            elements: {{ point: {{ radius: 0 }} }},
            scales: {{
                x: {{ display: false }},
                y: {{ ticks: {{ font: {{ size: 9 }} }} }}
            }}
        }};

        function createChart(id, datasets, scaleKey) {{
            const opts = JSON.parse(JSON.stringify(commonOptions));
            if (scaleKey && scales[scaleKey]) {{
                opts.scales.y.min = scales[scaleKey].min;
                opts.scales.y.max = scales[scaleKey].max;
            }}
            return new Chart(document.getElementById(id), {{
                type: 'line',
                data: {{ labels: [], datasets: datasets }},
                options: opts
            }});
        }}

        const charts = {{
            sog: createChart('chart_sog', [{{ label: 'SOG', data: [], borderColor: '#2563eb', borderWidth: 1.5 }}], 'sog'),
            heading: createChart('chart_heading', [
                {{ label: 'Heading', data: [], borderColor: '#059669', borderWidth: 1.5 }},
                {{ label: 'COG', data: [], borderColor: '#d97706', borderWidth: 1.5 }}
            ], 'heading'),
            depth: createChart('chart_depth', [{{ label: 'Depth', data: [], borderColor: '#0891b2', borderWidth: 1.5, fill: true, backgroundColor: 'rgba(8,145,178,0.1)' }}], 'depth'),
            wind: createChart('chart_wind', [{{ label: 'Wind', data: [], borderColor: '#7c3aed', borderWidth: 1.5 }}], 'wind'),
            rudder: createChart('chart_rudder', [{{ label: 'Rudder', data: [], borderColor: '#dc2626', borderWidth: 1.5 }}], 'rudder'),
            pitchroll: createChart('chart_pitchroll', [
                {{ label: 'Pitch', data: [], borderColor: '#0891b2', borderWidth: 1.5 }},
                {{ label: 'Roll', data: [], borderColor: '#ea580c', borderWidth: 1.5 }}
            ], 'pitchroll')
        }};

        // Logic
        function update() {{
            if (idx >= data.length) {{ togglePlay(); return; }}
            
            const d = data[idx];
            
            // Map
            marker.setLatLng([d.lat, d.lon]);
            trail.addLatLng([d.lat, d.lon]);
            if (idx % 50 === 0) map.panTo([d.lat, d.lon]);
            
            // Charts
            const pushData = (chart, vals) => {{
                chart.data.labels.push(idx);
                chart.data.datasets.forEach((ds, i) => ds.data.push(vals[i]));
                
                if (chart.data.labels.length > historyLen) {{
                    chart.data.labels.shift();
                    chart.data.datasets.forEach(ds => ds.data.shift());
                }}
                chart.update('none');
            }};
            
            pushData(charts.sog, [d.sog]);
            pushData(charts.heading, [d.heading, d.cog]);
            pushData(charts.depth, [d.depth]);
            pushData(charts.wind, [d.wind_speed]);
            pushData(charts.rudder, [d.rudder]);
            pushData(charts.pitchroll, [d.pitch, d.roll]);
            
            document.getElementById('idxDisplay').textContent = idx;
            document.getElementById('timeline').value = idx;
            idx++;
        }}

        function seek(val) {{
            const newIdx = parseInt(val);
            idx = newIdx;
            const d = data[idx];
            
            // Update Map
            marker.setLatLng([d.lat, d.lon]);
            map.panTo([d.lat, d.lon]);
            
            // Rebuild Trail
            const trailPoints = data.slice(0, idx+1).map(p => [p.lat, p.lon]);
            trail.setLatLngs(trailPoints);
            
            // Rebuild Charts
            const start = Math.max(0, idx - historyLen);
            const end = idx + 1;
            const slice = data.slice(start, end);
            const labels = slice.map((_, i) => start + i);
            
            const updateChart = (chart, keys) => {{
                chart.data.labels = labels;
                chart.data.datasets.forEach((ds, i) => {{
                    ds.data = slice.map(item => item[keys[i]]);
                }});
                chart.update('none');
            }};
            
            updateChart(charts.sog, ['sog']);
            updateChart(charts.heading, ['heading', 'cog']);
            updateChart(charts.depth, ['depth']);
            updateChart(charts.wind, ['wind_speed']);
            updateChart(charts.rudder, ['rudder']);
            updateChart(charts.pitchroll, ['pitch', 'roll']);
            
            document.getElementById('idxDisplay').textContent = idx;
        }}

        function togglePlay() {{
            playing = !playing;
            const btn = document.getElementById('playBtn');
            if (playing) {{
                btn.textContent = '⏸ Pause';
                timer = setInterval(update, speed);
            }} else {{
                btn.textContent = '▶ Play';
                clearInterval(timer);
            }}
        }}

        function reset() {{
            playing = false;
            clearInterval(timer);
            document.getElementById('playBtn').textContent = '▶ Play';
            idx = 0;
            document.getElementById('timeline').value = 0;
            trail.setLatLngs([]);
            Object.values(charts).forEach(c => {{
                c.data.labels = [];
                c.data.datasets.forEach(ds => ds.data = []);
                c.update();
            }});
            marker.setLatLng([data[0].lat, data[0].lon]);
            map.setView([data[0].lat, data[0].lon], 14);
            document.getElementById('idxDisplay').textContent = 0;
        }}

        function setSpeed(val) {{
            speed = 200 - ((val - 1) * (190/99));
            document.getElementById('speedVal').textContent = Math.round(1000/speed) + 'ups';
            if (playing) {{
                clearInterval(timer);
                timer = setInterval(update, speed);
            }}
        }}

        function toggleScale(key, btn) {{
            const chart = charts[key];
            const isFixed = btn.classList.contains('active');
            
            if (isFixed) {{
                delete chart.options.scales.y.min;
                delete chart.options.scales.y.max;
                btn.classList.remove('active');
                btn.textContent = 'Auto';
            }} else {{
                if (scales[key]) {{
                    chart.options.scales.y.min = scales[key].min;
                    chart.options.scales.y.max = scales[key].max;
                }}
                btn.classList.add('active');
                btn.textContent = 'Fixed';
            }}
            chart.update();
        }}
    </script>
</body>
</html>
"""

with open('Phase0/results/boat_grid.html', 'w') as f:
    f.write(html)

print("✓ Generated Phase0/results/boat_grid.html")
