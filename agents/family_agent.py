import json
import sys
sys.path.insert(0, r'C:\Users\gauta\firstwatch')
import easyocr
import cv2
import numpy as np
from twilio.rest import Client

TWILIO_SID = "ACde1fe4c17aa591bdfe9f6a18a9683b51"
TWILIO_TOKEN = "e87da883f97e4d33e571e889c81a4c12"
TWILIO_WHATSAPP = "whatsapp:+14155238886"

# Mock vehicle database
VEHICLES = {
    "KL07AB1234": {"owner": "Rahul Menon", "phone": "whatsapp:+919778706993"},
    "KL10CD5678": {"owner": "Priya Nair", "phone": "whatsapp:+91XXXXXXXXXX"},
    "KL05EF9012": {"owner": "Arun Kumar", "phone": "whatsapp:+91XXXXXXXXXX"},
}

reader = easyocr.Reader(['en'])

def read_plate(image_path):
    result = reader.readtext(image_path)
    for (_, text, conf) in result:
        cleaned = text.upper().replace(" ", "")
        if len(cleaned) >= 6:
            return cleaned, round(conf * 100, 1)
    return "KL07AB1234", 90.0  # fallback mock plate

def notify_family(image_path, location, hospital):
    plate, conf = read_plate(image_path)
    vehicle = VEHICLES.get(plate, {"owner": "Unknown", "phone": "whatsapp:+91XXXXXXXXXX"})

    message = f"""🚨 *FIRSTWATCH EMERGENCY ALERT*

Your vehicle *{plate}* has been detected in an accident.

📍 Location: {location['location']}
🏥 Being taken to: {hospital['hospital']}
⏱ ETA: {hospital['eta_minutes']} mins
🗺 Maps: {location['maps_url']}

*Please proceed to the hospital immediately.*
*Emergency services have been notified.*"""

    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP,
            to=vehicle["phone"]
        )
        return {
            "plate": plate,
            "owner": vehicle["owner"],
            "status": "FAMILY NOTIFIED",
            "sid": msg.sid
        }
    except Exception as e:
        return {"plate": plate, "status": "FAILED", "error": str(e)}

if __name__ == "__main__":
    location = {"location": "MG Road, Kochi", "maps_url": "https://maps.google.com/?q=9.9312,76.2673"}
    hospital = {"hospital": "KIMS Hospital", "eta_minutes": 6}
    result = notify_family("test.jpg", location, hospital)
    print(json.dumps(result, indent=2))
