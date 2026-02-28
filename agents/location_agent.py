import json

# This acts as Flink's internal database of registered CCTV cameras
CAMERA_DATABASE = {
    "CAM_001": {
        "lat": 10.0415, 
        "lng": 76.3243, 
        "location": "CUSAT Main Gate (Kalamassery)"
    },
    "CAM_002": {
        "lat": 10.0432, 
        "lng": 76.3211, 
        "location": "CUSAT Software Engineering Block"
    },
    "CAM_003": {
        "lat": 10.0261, 
        "lng": 76.3125, 
        "location": "Pathadipalam NH-66 Junction"
    }
}

def get_location(camera_id="CAM_001"):
    print(f"🌍 Database Lookup: Fetching registered coordinates for {camera_id}...")
    
    # Look up the camera in the database (defaults to CAM_001 at CUSAT if not found)
    cam_data = CAMERA_DATABASE.get(camera_id, CAMERA_DATABASE["CAM_001"])
    
    lat = cam_data["lat"]
    lng = cam_data["lng"]
    live_address = cam_data["location"]
    
    # Generate the clickable Google Maps link for the WhatsApp Agents
    maps_url = f"https://www.google.com/maps?q={lat},{lng}"
    
    return {
        "camera_id": camera_id,
        "lat": lat,
        "lng": lng,
        "location": live_address,
        "maps_url": maps_url
    }

if __name__ == "__main__":
    # Test it by asking for CAM_001 (CUSAT)
    print(json.dumps(get_location("CAM_001"), indent=2))