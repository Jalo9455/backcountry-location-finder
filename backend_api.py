from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter

app = Flask(__name__)

# Manual CORS handling
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Sample elevation profiles for known Colorado backcountry locations
# Each profile is a normalized array of elevations representing the skyline
LOCATION_DATABASE = {
    'Berthoud Pass': {
        'elevation': 11307,
        'coordinates': (39.7983, -105.7783),
        'aspect': 'North',
        'terrain': 'Alpine bowl with steep chutes',
        'profile': np.array([0.3, 0.35, 0.4, 0.5, 0.65, 0.8, 0.85, 0.75, 0.6, 0.45, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.1, 0.15, 0.2]),
        'viewpoint_elevation': 11307
    },
    'Loveland Pass': {
        'elevation': 11990,
        'coordinates': (39.6656, -105.8783),
        'aspect': 'East',
        'terrain': 'Steep alpine with multiple gullies',
        'profile': np.array([0.2, 0.25, 0.4, 0.55, 0.7, 0.85, 0.9, 0.95, 0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.15, 0.1, 0.08, 0.12, 0.18, 0.25]),
        'viewpoint_elevation': 11990
    },
    'Jones Pass': {
        'elevation': 12451,
        'coordinates': (39.7917, -105.8833),
        'aspect': 'Northeast',
        'terrain': 'Wide alpine bowl',
        'profile': np.array([0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.7, 0.65, 0.55, 0.45, 0.35, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.15, 0.2, 0.3]),
        'viewpoint_elevation': 12451
    },
    'Torreys Peak': {
        'elevation': 14267,
        'coordinates': (39.6428, -105.8211),
        'aspect': 'North',
        'terrain': 'Fourteener with steep couloirs',
        'profile': np.array([0.1, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 0.95, 0.8, 0.65, 0.5, 0.35, 0.25, 0.2, 0.15, 0.1, 0.08, 0.1, 0.15]),
        'viewpoint_elevation': 14267
    },
    'A-Basin West Wall': {
        'elevation': 13050,
        'coordinates': (39.6422, -105.8717),
        'aspect': 'East',
        'terrain': 'Steep chutes and bowls',
        'profile': np.array([0.25, 0.3, 0.45, 0.6, 0.75, 0.85, 0.8, 0.7, 0.55, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1, 0.12, 0.15, 0.2]),
        'viewpoint_elevation': 12500
    },
    'Mount Sniktau': {
        'elevation': 13234,
        'coordinates': (39.6719, -105.8850),
        'aspect': 'Northeast',
        'terrain': 'Rolling alpine with cornices',
        'profile': np.array([0.35, 0.4, 0.5, 0.6, 0.7, 0.75, 0.72, 0.65, 0.55, 0.45, 0.38, 0.32, 0.28, 0.25, 0.22, 0.18, 0.15, 0.18, 0.22, 0.28]),
        'viewpoint_elevation': 11990
    },
    'Herman Gulch': {
        'elevation': 11900,
        'coordinates': (39.7039, -105.8342),
        'aspect': 'East',
        'terrain': 'Tree-lined gulch with open slopes',
        'profile': np.array([0.3, 0.35, 0.45, 0.55, 0.65, 0.7, 0.68, 0.6, 0.5, 0.42, 0.35, 0.3, 0.26, 0.22, 0.19, 0.16, 0.14, 0.16, 0.2, 0.26]),
        'viewpoint_elevation': 10500
    },
    'Mayflower Gulch': {
        'elevation': 12100,
        'coordinates': (39.4639, -106.0839),
        'aspect': 'North',
        'terrain': 'Alpine cirque with moderate slopes',
        'profile': np.array([0.32, 0.38, 0.48, 0.58, 0.68, 0.73, 0.7, 0.63, 0.53, 0.43, 0.36, 0.31, 0.27, 0.24, 0.21, 0.18, 0.16, 0.18, 0.22, 0.28]),
        'viewpoint_elevation': 11000
    },
    'Quandary Peak': {
        'elevation': 14265,
        'coordinates': (39.3972, -106.1064),
        'aspect': 'East',
        'terrain': 'Fourteener with steep east face',
        'profile': np.array([0.12, 0.18, 0.32, 0.52, 0.72, 0.87, 0.96, 0.98, 0.92, 0.78, 0.62, 0.48, 0.36, 0.27, 0.22, 0.18, 0.14, 0.12, 0.14, 0.18]),
        'viewpoint_elevation': 14265
    },
    'Butler Gulch': {
        'elevation': 12200,
        'coordinates': (39.7453, -105.8456),
        'aspect': 'Northeast',
        'terrain': 'Wide gulch with mellow terrain',
        'profile': np.array([0.28, 0.33, 0.42, 0.52, 0.62, 0.68, 0.65, 0.58, 0.49, 0.41, 0.34, 0.29, 0.25, 0.22, 0.19, 0.17, 0.15, 0.17, 0.21, 0.26]),
        'viewpoint_elevation': 10500
    }
}

def extract_skyline(image_array):
    """
    Extract skyline profile from image using edge detection.
    Returns normalized array of elevations.
    """
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2)
    else:
        gray = image_array
    
    # Apply Sobel edge detection (vertical edges)
    edges = ndimage.sobel(gray, axis=0)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Extract skyline: find highest strong edge in each column
    skyline = np.zeros(width)
    threshold = np.percentile(np.abs(edges), 70)
    
    for col in range(width):
        # Look for edges in upper 70% of image
        search_height = int(height * 0.7)
        column_edges = np.abs(edges[:search_height, col])
        
        # Find highest strong edge
        strong_edges = np.where(column_edges > threshold)[0]
        if len(strong_edges) > 0:
            skyline[col] = strong_edges[0]
        else:
            skyline[col] = 0
    
    # Smooth the skyline
    if len(skyline) > 10:
        window_length = min(11, len(skyline) - 1 if len(skyline) % 2 == 0 else len(skyline))
        skyline = savgol_filter(skyline, window_length, 3)
    
    # Normalize to 0-1 range
    if skyline.max() > skyline.min():
        skyline = (skyline - skyline.min()) / (skyline.max() - skyline.min())
    
    return skyline

def compare_profiles(profile1, profile2):
    """
    Compare two skyline profiles and return similarity score (0-100).
    Uses correlation and RMSE.
    """
    # Interpolate to same length
    target_length = 100
    x1 = np.linspace(0, 1, len(profile1))
    x2 = np.linspace(0, 1, len(profile2))
    x_new = np.linspace(0, 1, target_length)
    
    p1_interp = np.interp(x_new, x1, profile1)
    p2_interp = np.interp(x_new, x2, profile2)
    
    # Calculate correlation
    correlation = np.corrcoef(p1_interp, p2_interp)[0, 1]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((p1_interp - p2_interp) ** 2))
    
    # Combine metrics (correlation weighted more)
    similarity = (correlation * 0.7 + (1 - rmse) * 0.3) * 100
    
    return max(0, min(100, similarity))

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Backcountry Ski Location Finder API',
        'endpoints': {
            '/analyze': 'POST - Analyze photo and return location matches'
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Extract skyline from uploaded image
        uploaded_skyline = extract_skyline(image_array)
        
        # Compare with database
        matches = []
        for location_name, location_data in LOCATION_DATABASE.items():
            similarity = compare_profiles(uploaded_skyline, location_data['profile'])
            
            matches.append({
                'location': location_name,
                'confidence': round(similarity, 1),
                'elevation': location_data['elevation'],
                'coordinates': location_data['coordinates'],
                'aspect': location_data['aspect'],
                'terrain': location_data['terrain']
            })
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return top 5 matches
        return jsonify({
            'success': True,
            'matches': matches[:5]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
