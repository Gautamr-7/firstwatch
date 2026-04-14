#  Project Echo: Autonomous Crowd-Sourced Triage Engine

[](https://www.google.com/search?q=https://www.python.org/)
[](https://www.google.com/search?q=https://www.twilio.com/)
[](https://www.google.com/search?q=https://streamlit.io/)
[](https://www.google.com/search?q=https://ultralytics.com/)

**Turning "Bystander Voyeurism" into a Life-Saving Sensor Network.** Built for INNOBOT 2.0 (Healthcare & Smart Living Track).


##  The Problem

During severe accidents on Indian roads, the emergency response system faces two massive bottlenecks:

1.  **The Call Center Collapse:** Mass-casualty events or busy highway crashes generate dozens of duplicate, panicked calls that crash 108 switchboards and delay dispatch.
2.  **The Data Black Hole:** Bystanders pull out their smartphones to record videos for social media instead of helping. Millions of HD cameras are present at every accident scene, but zero pipeline exists to use that data to triage the victim.

##  The Solution

**Project Echo** is a WhatsApp-native, multi-agent ingestion engine that requires zero new government infrastructure.

When a crash happens, bystanders simply send their photos/videos and their WhatsApp Live Location to the Echo Emergency Bot. The system ingests simultaneous reports, autonomously clusters them to prevent duplicate ambulance dispatches, validates the threat severity using Computer Vision, and instantly dispatches the nearest ambulance while broadcasting AI-generated physical instructions back to the crowd.

-----

##  System Architecture

Echo operates on a 3-stage autonomous Multi-Agent Pipeline:

1.  **Validation Agent (Threat Assessment):** - Ingests incoming media and parses EXIF/metadata.

      - Runs the frame through a custom **YOLOv8 model** to validate the threat (e.g., *Severe Front-Impact, Fire detected, Pedestrian involved*).

2.  **Geospatial Clustering Agent (The Deduplicator):**

      - Merges simultaneous incoming WhatsApp messages based on GPS radius and timestamps.
      - Example: 15 messages from a 50m radius within 2 minutes are collapsed into a single `Incident ID`, preventing the dispatch of 15 separate ambulances to the same crash.

3.  **Tactical Dispatch Agent (Broker):**

      - Hits the Hospital Dashboard webhook to autonomously dispatch the nearest Advanced Life Support (ALS) unit.
      - Executes a mass-broadcast reply via Twilio to every bystander who messaged the bot, providing specific crowd-control instructions based on YOLOv8's assessment.

-----

##  Tech Stack

  * **Core Language:** Python 3.10+
  * **Computer Vision:** YOLOv8 (Ultralytics), Roboflow (Dataset & Pre-training)
  * **Agentic Orchestration:** LangChain / Custom Python Pipeline
  * **Messaging & Communications:** Twilio WhatsApp Business API
  * **Frontend Dashboard:** Streamlit
  * **Geospatial Processing:** Geopy / Haversine formula (for clustering)

-----

##  Key Features

  - **Zero Infrastructure Required:** Leverages the 800 million smartphones already in Indian pockets rather than waiting for Smart City CCTV installations.
  - **Flawless Deduplication:** Converts the chaos of a panicked crowd into a single, highly accurate 360-degree digital twin of the crash site.
  - **Real Autonomous Action:** Does not just act as a dashboard. It makes physical world interventions (dispatching vehicles and executing crowd control) without human bottlenecking.
  - **Zero-Friction UX:** Bystanders do not need to download an SOS app. They use WhatsApp, an app they already have open.

-----

##  Installation & Setup

### Prerequisites

  - Python 3.10 or higher
  - A Twilio Developer Account (with WhatsApp Sandbox configured)
  - Ngrok (for local webhook testing)

### Step-by-Step Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/project-echo.git
    cd project-echo
    ```

2.  **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables:**
    Create a `.env` file in the root directory and add your Twilio credentials:

    ```env
    TWILIO_ACCOUNT_SID=your_account_sid
    TWILIO_AUTH_TOKEN=your_auth_token
    TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
    ```

5.  **Run the Streamlit Dashboard:**

    ```bash
    streamlit run app.py
    ```

6.  **Start the Webhook Server (Ngrok):**

    ```bash
    ngrok http 8501
    ```

    *Copy the generated HTTPS URL and paste it into your Twilio WhatsApp Sandbox configuration under "When a message comes in".*

-----

##  Usage Workflow

1.  **The Incident:** Open the Streamlit dashboard (`localhost:8501`) to view the Central Dispatch map.
2.  **Crowd Ingestion:** Send a WhatsApp message with an image of a car crash and a location pin to the Twilio Sandbox number.
3.  **Observation:** Watch the Streamlit dashboard instantly log the incoming data, validate it via YOLOv8, and plot it on the map.
4.  **Autonomous Action:** Send 2-3 more messages from different numbers. Watch the system cluster them into a single incident and automatically fire a WhatsApp response back to your phone with tactical instructions.



*If this project saves even one life by cutting through the noise, it has done its job.*
