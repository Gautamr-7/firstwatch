import json
from twilio.rest import Client

TWILIO_SID = "ACde1fe4c17aa591bdfe9f6a18a9683b51"
TWILIO_TOKEN = "e87da883f97e4d33e571e889c81a4c12"
TWILIO_WHATSAPP = "whatsapp:+14155238886"
YOUR_PHONE = "whatsapp:+919778706993"  # your phone number with country code

def dispatch_ambulance(location, hospital, severity, ambulances):
    message = f"""🚨 *FIRSTWATCH ALERT*
    
*ACCIDENT DETECTED*
📍 Location: {location['location']}
🔴 Severity: {severity}
🚑 Ambulances Required: {ambulances}
🏥 Hospital: {hospital['hospital']}
⏱ ETA: {hospital['eta_minutes']} mins
🗺 Maps: {location['maps_url']}

*Dispatch immediately.*"""

    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP,
            to=YOUR_PHONE
        )
        return {"status": "SENT", "sid": msg.sid, "message": message}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}

if __name__ == "__main__":
    location = {"location": "MG Road, Kochi", "maps_url": "https://maps.google.com/?q=9.9312,76.2673"}
    hospital = {"hospital": "KIMS Hospital", "eta_minutes": 6}
    result = dispatch_ambulance(location, hospital, "CRITICAL", 2)
    print(json.dumps(result, indent=2))