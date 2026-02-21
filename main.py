import asyncio
import base64
import json
import logging
import os
from typing import Optional

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Correct import - this is the official one
import google.generativeai as genai
from google.generativeai.types import Blob

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("receptionist")

app = FastAPI(root_path="/")

print("Routes registered:")
for route in app.routes:
    print(f"  {route.path} {route.methods}")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
N8N_CALL_START_URL = os.environ.get("N8N_CALL_START_URL")
N8N_BOOK_APPOINTMENT_URL = os.environ.get("N8N_BOOK_APPOINTMENT_URL")
N8N_POST_CALL_URL = os.environ.get("N8N_POST_CALL_URL")

GEMINI_MODEL = "gemini-2.0-flash-exp"

# Configure the client properly
genai.configure(api_key=GEMINI_API_KEY)

session_store: dict[str, dict] = {}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    log.info("HIT /incoming-call endpoint!")
    form = await request.form()
    log.debug(f"Form data received: {form}")

    caller_phone = form.get("From", "")
    called_number = form.get("To", "")
    call_sid = form.get("CallSid", "")

    log.info(f"Incoming call | from={caller_phone} to={called_number} sid={call_sid}")

    client_config = await lookup_client(caller_phone, called_number, call_sid)

    if not client_config:
        log.error("Failed to retrieve client config from n8n.")
        return HTMLResponse(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Connection error. Please try again later.</Say><Hangup/></Response>',
            media_type="text/xml",
        )

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

    ws_host = os.environ.get("WEBSOCKET_HOST", request.headers.get("host", "localhost"))
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{ws_host}/ws">
      <Parameter name="callSid" value="{call_sid}"/>
    </Stream>
  </Connect>
</Response>"""
    log.info(f"Returning TwiML with WS URL: wss://{ws_host}/ws")
    return HTMLResponse(content=twiml, media_type="text/xml")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    log.info("WebSocket connection attempt received")
    await ws.accept()
    log.info("WebSocket accepted")

    call_sid = None
    stream_sid = None

    try:
        initial_msg = await ws.receive_text()
        msg = json.loads(initial_msg)

        if msg.get("event") == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid = msg["start"].get("customParameters", {}).get("callSid")
            log.info(f"Stream started | sid={call_sid} | stream={stream_sid}")

            session = session_store.get(call_sid, {})

            config = {
                "response_modalities": ["AUDIO"],
                "system_instruction": session.get("systemPrompt"),
                "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}},
            }

            async with genai.GenerativeModel(GEMINI_MODEL).start_chat() as chat:  # Use chat for simplicity
                log.info("Gemini chat session started")

                # Send initial greeting as text
                greeting = "Hi, this is Ava with Sunlight Solar. To get started, are you the home owner?"
                response = chat.send_message(greeting)
                log.info(f"Gemini text response: {response.text}")

                # For audio, we'd need to convert text to speech - add gTTS here if needed

                async def twilio_to_gemini():
                    try:
                        async for message in _websocket_stream(ws):
                            if message.get("event") == "media":
                                payload = base64.b64decode(message["media"]["payload"])
                                pcm_8k = audioop.ulaw2lin(payload, 2)
                                pcm_24k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)
                                # Send audio blob to Gemini (if using audio input mode)
                                # chat.send_message(Blob(data=pcm_24k, mime_type="audio/pcm;rate=24000"))
                                log.debug("Audio chunk sent to Gemini")
                            elif message.get("event") == "stop":
                                break
                    except Exception as e:
                        log.error(f"Twilio->Gemini error: {e}")

                async def gemini_to_twilio():
                    try:
                        # This would be the audio response loop - simplified for now
                        log.info("Gemini to Twilio loop started")
                        # You'd receive audio from Gemini here and convert/send back
                    except Exception as e:
                        log.error(f"Gemini->Twilio error: {e}")

                await asyncio.gather(twilio_to_gemini(), gemini_to_twilio())

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected | sid={call_sid}")
    finally:
        if call_sid:
            await trigger_post_call(call_sid)

async def lookup_client(from_number, to_number, sid):
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
    session = session_store.get(call_sid, {})
    payload = {**args, "callSid": call_sid, "calendarId": session.get("calendarId")}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(N8N_BOOK_APPOINTMENT_URL, json=payload)
            return resp.json()
    except Exception:
        return {"success": False, "error": "Booking failed"}

async def trigger_post_call(call_sid):
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
