import json
with open("state.json", "w") as f:
    json.dump({"accident": False, "severity": 0, "dispatch_time": "", "crash_timestamp": 0}, f)
print("✅ Dashboard reset to GREEN. Ready for the next judge!")
