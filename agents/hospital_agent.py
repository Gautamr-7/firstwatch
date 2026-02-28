import json

HOSPITALS = [
    {"name": "Kinder Hospital", "lat": 10.0403, "lng": 76.3214, "trauma": True, "icu_beds": 5, "phone": "+919778706993"},
    {"name": "Rajagiri Hospital", "lat": 9.9986, "lng": 76.2999, "trauma": True, "icu_beds": 5, "phone": "+914842524400"},
    {"name": "Lakeshore Hospital", "lat": 9.9638, "lng": 76.2973, "trauma": True, "icu_beds": 3, "phone": "+914842701032"},
    {"name": "KIMS Hospital", "lat": 9.9312, "lng": 76.2673, "trauma": True, "icu_beds": 7, "phone": "+914842805000"},
    {"name": "Aster Medcity", "lat": 9.9816, "lng": 76.2954, "trauma": True, "icu_beds": 2, "phone": "+914842676000"},
]
def get_nearest_hospital(lat, lng, severity):
    import math

    def distance(h):
        return math.sqrt((h["lat"] - lat) ** 2 + (h["lng"] - lng) ** 2)

    available = [h for h in HOSPITALS if h["trauma"] and h["icu_beds"] > 0]
    nearest = min(available, key=distance)

    # Mark bed as reserved
    nearest["icu_beds"] -= 1

    return {
        "hospital": nearest["name"],
        "phone": nearest["phone"],
        "icu_beds_remaining": nearest["icu_beds"],
        "distance_km": round(distance(nearest) * 111, 2),
        "eta_minutes": round(distance(nearest) * 111 * 2, 0),
        "status": "BED CONFIRMED"
    }

if __name__ == "__main__":
    result = get_nearest_hospital(9.9312, 76.2673, "CRITICAL")
    print(json.dumps(result, indent=2))
