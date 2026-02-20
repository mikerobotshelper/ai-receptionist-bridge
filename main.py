import asyncio
import base64
import json
import logging
import os
from typing import Optional

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

# --- Audio Compatibility for Python 3.13+ ---
try:
    import audioop
except ImportError:
    import audioop_lts as audioop

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("receptionist")

# --- Environment Variables ---
# Ensure these are set in your Railway 'Variables' tab
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
N8N_CALL_START_URL = os.environ.get("N8N_CALL_START_URL")
N8N_BOOK_APPOINTMENT_URL = os.environ.get("N8N_BOOK_APPOINTMENT_URL")
N8N_POST_CALL_URL = os.environ.get("N8N_POST_CALL_URL")

# Model: Using the 2026 stable Gemini Live model
GEMINI_MODEL = "gemini-2.0-flash-exp"

# --- Gemini Client ---
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

# Store session data in memory (CallSid as key)
session_store: dict[str, dict] = {}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Step 1: Twilio calls this endpoint. 
    We fetch client data from n8n and return the TwiML <Connect><Stream> to Twilio.
    """
    form = await request.form()
    caller_phone = form.get("From", "")
    called_number = form.get("To", "")
    call_sid = form.get("CallSid", "")
    
    log.info(f"Incoming call | from={caller_phone} to={called_number} sid={call_sid}")

    # Fetch configuration (Flow A)
    client_config = await lookup_client(caller_phone, called_number, call_sid)
    
    if not client_config:
        log.error("Failed to retrieve client config from n8n.")
        return HTMLResponse(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Connection error. Please try again later.</Say><Hangup/></Response>',
            media_type="text/xml",
        )

    # Store session details for use during the call
    session_store[call_sid] = {
        "callerPhone": caller_phone,
        "calledNumber": called_number,
        "companyName": client_config.get("companyName", "Our Business"),
        "calendarId": client_config.get("calendarId", ""),
        "timezone": client_config.get("timezone", "UTC"),
        "systemPrompt": client_config.get("systemPrompt", "You are a helpful AI receptionist."),
        "clientRecordId": client_config.get("clientRecordId", ""),
        "appointmentBooked": False
    }

    host = request.headers.get("host", "")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/ws">
      <Parameter name="callSid" value="{call_sid}"/>
    </Stream>
  </Connect>
</Response>"""
    return HTMLResponse(content=twiml, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Step 2: The WebSocket bridge between Twilio and Gemini.
    """
    await ws.accept()
    call_sid = None
    stream_sid = None
    
    try:
        # Wait for Twilio 'start' message to identify the call
        initial_msg = await ws.receive_text()
        msg = json.loads(initial_msg)
        
        if msg.get("event") == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid = msg["start"].get("customParameters", {}).get("callSid")
            log.info(f"Stream started | sid={call_sid} | stream={stream_sid}")

        session = session_store.get(call_sid, {})
        
        # Configure Gemini Live Session
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": session.get("systemPrompt"),
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}},
            "tools": [_get_tools_config()],
        }

        async with gemini_client.aio.live.connect(model=GEMINI_MODEL, config=config) as gemini_session:
            
            async def twilio_to_gemini():
                """Reads audio from Twilio and sends to Gemini."""
                try:
                    async for message in _websocket_stream(ws):
                        if message.get("event") == "media":
                            payload = base64.b64decode(message["media"]["payload"])
                            # Convert 8kHz mu-law to 24kHz PCM for Gemini
                            pcm_8k = audioop.ulaw2lin(payload, 2)
                            pcm_24k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)
                            await gemini_session.send(input=types.Blob(data=pcm_24k, mime_type="audio/pcm;rate=24000"))
                        elif message.get("event") == "stop":
                            break
                except Exception as e:
                    log.error(f"Twilio->Gemini Error: {e}")

            async def gemini_to_twilio():
                """Reads responses from Gemini and sends audio/tool-calls to Twilio/n8n."""
                try:
                    async for response in gemini_session.receive():
                        if response.data:
                            # Convert 24kHz PCM to 8kHz mu-law for Twilio
                            pcm_8k, _ = audioop.ratecv(response.data, 2, 1, 24000, 8000, None)
                            mulaw = audioop.lin2ulaw(pcm_8k, 2)
                            await ws.send_text(json.dumps({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": base64.b64encode(mulaw).decode("utf-8")},
                            }))
                        
                        if response.tool_call:
                            for fc in response.tool_call.function_calls:
                                # Handle Flow B (Booking)
                                tool_result = await handle_booking(fc.args, call_sid)
                                await gemini_session.send(input=types.LiveClientToolResponse(
                                    function_responses=[types.FunctionResponse(
                                        id=fc.id, name=fc.name, response=tool_result
                                    )]
                                ))
                except Exception as e:
                    log.error(f"Gemini->Twilio Error: {e}")

            await asyncio.gather(twilio_to_gemini(), gemini_to_twilio())

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected | sid={call_sid}")
    finally:
        if call_sid:
            # Trigger Flow C (Post-call logging/email)
            await trigger_post_call(call_sid)

# --- Helper Functions ---

async def lookup_client(from_number, to_number, sid):
    """Hits n8n Flow A for client business data."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(N8N_CALL_START_URL, json={
                "callerPhone": from_number,
                "calledNumber": to_number,
                "callSid": sid
            })
            return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None

async def handle_booking(args, call_sid):
    """Hits n8n Flow B to book an appointment."""
    session = session_store.get(call_sid, {})
    payload = {**args, "callSid": call_sid, "calendarId": session.get("calendarId")}
    
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(N8N_BOOK_APPOINTMENT_URL, json=payload)
            res_data = resp.json()
            if res_data.get("success"):
                session_store[call_sid]["appointmentBooked"] = True
            return res_data
    except Exception as e:
        return {"success": False, "error": str(e)}

async def trigger_post_call(call_sid):
    """Hits n8n Flow C for call summary and email."""
    session = session_store.pop(call_sid, {})
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(N8N_POST_CALL_URL, json=session)
    except Exception:
        pass

async def _websocket_stream(ws):
    while True:
        try:
            yield json.loads(await ws.receive_text())
        except:
            break

def _get_tools_config():
    return types.Tool(function_declarations=[types.FunctionDeclaration(
        name="book_appointment",
        description="Book an appointment by collecting name, email, date, and time.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "name": types.Schema(type=types.Type.STRING),
                "email": types.Schema(type=types.Type.STRING),
                "date": types.Schema(type=types.Type.STRING),
                "time": types.Schema(type=types.Type.STRING),
            },
            required=["name", "email", "date", "time"]
        )
    )])

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
