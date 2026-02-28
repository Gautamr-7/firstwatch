import json

CAMERAS = {
    "CAM_001": {"lat": 9.9312, "lng": 76.2673, "location": "MG Road, Kochi, Kerala"},
    "CAM_002": {"lat": 9.9816, "lng": 76.2999, "location": "Edappally Junction, Kochi"},
    "CAM_003": {"lat": 8.5241, "lng": 76.9366, "location": "Palayam, Thiruvananthapuram"},
}

def get_location(camera_id="CAM_001"):
    cam = CAMERAS.get(camera_id, CAMERAS["CAM_001"])
    return {
        "camera_id": camera_id,
        "lat": cam["lat"],
        "lng": cam["lng"],
        "location": cam["location"],
        "maps_url": f"https://maps.google.com/?q={cam['lat']},{cam['lng']}"
    }

if __name__ == "__main__":
    print(json.dumps(get_location("CAM_001"), indent=2))
