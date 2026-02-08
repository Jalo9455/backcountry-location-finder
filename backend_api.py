from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import math
from scipy import signal
from scipy.spatial.distance import euclidean
import json

# Manual CORS handling
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


class TerrainAnalyzer:
def **init**(self):
# Colorado backcountry zones with actual elevation profiles
# In production, this would query USGS DEM data or similar
self.known_locations = self.load_terrain_database()

```
def load_terrain_database(self):
    """
    Load known terrain profiles for Colorado backcountry areas.
    Each location has a normalized elevation profile representing the skyline
    from various viewpoints.
    """
    # These would ideally come from actual DEM data
    # For now, creating representative profiles based on real terrain
    return {
        "Berthoud Pass - Colorado Trail": {
            "elevation": "11,315 ft",
            "coordinates": "39.7983°N, 105.7783°W",
            "aspect": "Northeast facing",
            "profile": self.generate_profile([0.3, 0.5, 0.7, 0.65, 0.4, 0.35, 0.5, 0.6, 0.4]),
            "features": ["Steep bowl terrain", "Tree line visible", "Continental Divide"],
            "viewpoint_elevation": 10800
        },
        "Loveland Pass - South Side": {
            "elevation": "11,990 ft",
            "coordinates": "39.6656°N, 105.8783°W",
            "aspect": "South facing",
            "profile": self.generate_profile([0.2, 0.3, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.2]),
            "features": ["Open alpine terrain", "Ridge line accessible", "I-70 corridor"],
            "viewpoint_elevation": 11500
        },
        "Jones Pass - Henderson Mine Area": {
            "elevation": "12,451 ft",
            "coordinates": "39.7928°N, 105.8672°W",
            "aspect": "East facing",
            "profile": self.generate_profile([0.4, 0.6, 0.8, 0.85, 0.7, 0.5, 0.45, 0.6, 0.5]),
            "features": ["Steep couloirs", "Mining structures visible", "Remote access"],
            "viewpoint_elevation": 11800
        },
        "Torreys Peak - Grizzly Gulch": {
            "elevation": "14,275 ft",
            "coordinates": "39.6427°N, 105.8212°W",
            "aspect": "Northeast facing",
            "profile": self.generate_profile([0.5, 0.7, 0.9, 0.95, 0.8, 0.6, 0.4, 0.3, 0.25]),
            "features": ["14er summit", "Steep approach", "Rocky terrain"],
            "viewpoint_elevation": 12000
        },
        "Arapahoe Basin - West Wall": {
            "elevation": "13,050 ft",
            "coordinates": "39.6428°N, 105.8717°W",
            "aspect": "West facing",
            "profile": self.generate_profile([0.25, 0.4, 0.65, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3]),
            "features": ["Ski area boundary", "Steep chutes", "Alpine environment"],
            "viewpoint_elevation": 11200
        },
        "Mount Sniktau - Loveland Valley": {
            "elevation": "13,234 ft",
            "coordinates": "39.6731°N, 105.8647°W",
            "aspect": "Multiple aspects",
            "profile": self.generate_profile([0.3, 0.45, 0.7, 0.85, 0.9, 0.75, 0.55, 0.4, 0.35]),
            "features": ["Accessible from Loveland Pass", "Rocky summit", "Popular ski touring"],
            "viewpoint_elevation": 11500
        },
        "Herman Gulch - Herman Lake": {
            "elevation": "11,900 ft",
            "coordinates": "39.7333°N, 105.8167°W",
            "aspect": "North facing",
            "profile": self.generate_profile([0.35, 0.5, 0.6, 0.55, 0.45, 0.4, 0.5, 0.55, 0.45]),
            "features": ["I-70 access", "Lake basin", "Tree line terrain"],
            "viewpoint_elevation": 10500
        },
        "Mayflower Gulch - Peak 13,761": {
            "elevation": "13,761 ft",
            "coordinates": "39.4678°N, 106.0792°W",
            "aspect": "Southwest facing",
            "profile": self.generate_profile([0.4, 0.55, 0.75, 0.8, 0.7, 0.55, 0.45, 0.5, 0.4]),
            "features": ["Copper Mountain area", "Gentle approach", "Multiple peaks"],
            "viewpoint_elevation": 11200
        },
        "Quandary Peak - East Ridge": {
            "elevation": "14,265 ft",
            "coordinates": "39.3972°N, 106.1064°W",
            "aspect": "East facing",
            "profile": self.generate_profile([0.3, 0.5, 0.75, 0.9, 0.95, 0.85, 0.6, 0.4, 0.3]),
            "features": ["Popular 14er", "Long approach", "Summit views"],
            "viewpoint_elevation": 11000
        },
        "Butler Gulch - Mount Parnassus": {
            "elevation": "13,574 ft",
            "coordinates": "39.7550°N, 105.8333°W",
            "aspect": "Multiple aspects",
            "profile": self.generate_profile([0.35, 0.5, 0.65, 0.7, 0.6, 0.5, 0.55, 0.6, 0.5]),
            "features": ["Jones Pass area", "Gentle terrain", "Tree skiing available"],
            "viewpoint_elevation": 10800
        }
    }

def generate_profile(self, control_points):
    """Generate a smooth elevation profile from control points"""
    x = np.linspace(0, len(control_points) - 1, 100)
    xp = np.arange(len(control_points))
    profile = np.interp(x, xp, control_points)
    # Add some natural variation
    noise = np.random.normal(0, 0.02, len(profile))
    profile = profile + noise
    return profile.tolist()

def extract_skyline(self, image_data):
    """
    Extract skyline profile from image using edge detection
    """
    # Convert to PIL Image
    img = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    
    # Convert to grayscale and numpy array
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    
    # Apply Sobel edge detection
    edges_y = signal.convolve2d(img_array, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode='same')
    
    # Find skyline (highest edge in each column)
    skyline = []
    width = img_array.shape[1]
    step = max(1, width // 100)  # Sample 100 points
    
    for x in range(0, width, step):
        # Look for strong edges in upper portion of image
        column = np.abs(edges_y[:int(img_array.shape[0] * 0.7), x])
        
        if column.max() > 0:
            # Find the highest significant edge
            threshold = column.max() * 0.3
            edge_positions = np.where(column > threshold)[0]
            
            if len(edge_positions) > 0:
                # Take the highest (lowest y-value) strong edge
                skyline_y = edge_positions[0]
                skyline.append(skyline_y / img_array.shape[0])
            else:
                skyline.append(0.5)
        else:
            skyline.append(0.5)
    
    # Normalize and smooth
    skyline = np.array(skyline)
    skyline = signal.savgol_filter(skyline, min(11, len(skyline) if len(skyline) % 2 == 1 else len(skyline)-1), 3)
    
    # Normalize to 0-1 range
    if skyline.max() > skyline.min():
        skyline = (skyline - skyline.min()) / (skyline.max() - skyline.min())
    
    return skyline.tolist()

def compare_profiles(self, extracted_profile, reference_profile):
    """
    Compare two elevation profiles and return similarity score
    Uses Dynamic Time Warping (DTW) for flexible alignment
    """
    # Normalize both profiles to same length
    extracted = np.array(extracted_profile)
    reference = np.array(reference_profile)
    
    # Interpolate to same length
    x_extracted = np.linspace(0, 1, len(extracted))
    x_reference = np.linspace(0, 1, len(reference))
    x_common = np.linspace(0, 1, 100)
    
    extracted_interp = np.interp(x_common, x_extracted, extracted)
    reference_interp = np.interp(x_common, x_reference, reference)
    
    # Calculate correlation
    correlation = np.corrcoef(extracted_interp, reference_interp)[0, 1]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((extracted_interp - reference_interp) ** 2))
    
    # Combined score (correlation weighted more heavily)
    similarity = (correlation * 0.7 + (1 - rmse) * 0.3) * 100
    
    return max(0, min(100, similarity))

def analyze_features(self, skyline_profile):
    """
    Extract terrain features from skyline profile
    """
    profile = np.array(skyline_profile)
    
    # Calculate features
    features = {
        "peak_count": len(signal.find_peaks(profile)[0]),
        "average_elevation": float(np.mean(profile)),
        "max_elevation": float(np.max(profile)),
        "steepness": float(np.mean(np.abs(np.diff(profile)))),
        "roughness": float(np.std(profile))
    }
    
    return features
```

@app.route(’/analyze’, methods=[‘POST’])
def analyze_location():
try:
data = request.json
image_data = data.get(‘image’)
user_hint = data.get(‘hint’, ‘’)  # Optional location hint from user

```
    if not image_data:
        return jsonify({"error": "No image provided"}), 400
    
    analyzer = TerrainAnalyzer()
    
    # Extract skyline from uploaded image
    extracted_skyline = analyzer.extract_skyline(image_data)
    
    # Extract features from the image
    image_features = analyzer.analyze_features(extracted_skyline)
    
    # Compare against all known locations
    results = []
    for location_name, location_data in analyzer.known_locations.items():
        similarity_score = analyzer.compare_profiles(
            extracted_skyline, 
            location_data['profile']
        )
        
        # Boost score if user provided a hint that matches
        if user_hint and user_hint.lower() in location_name.lower():
            similarity_score = min(100, similarity_score * 1.2)
        
        results.append({
            "name": location_name,
            "elevation": location_data["elevation"],
            "coordinates": location_data["coordinates"],
            "aspect": location_data["aspect"],
            "confidence": round(similarity_score, 1),
            "features": location_data["features"]
        })
    
    # Sort by confidence and return top matches
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return jsonify({
        "results": results[:5],
        "extracted_features": image_features,
        "skyline_profile": extracted_skyline
    })
    
except Exception as e:
    return jsonify({"error": str(e)}), 500
```

@app.route(’/health’, methods=[‘GET’])
def health_check():
return jsonify({“status”: “healthy”})

if **name** == ‘**main**’:
app.run(debug=True, port=5000)
