import asyncio
import base64
import json
import logging
import os
from io import BytesIO
from typing import Optional

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# New official SDK (2026)
from google import genai
from google.genai import types

# TTS fallback
from gtts import gTTS
from pydub import AudioSegment

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

logging.basicConfig(level=logging.DEBUG)
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

# Configure new SDK
genai.configure(api_key=GEMINI_API_KEY)

session_store: dict[str, dict] = {}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    try:
        log.info("HIT /incoming-call endpoint!")
        form = await request.form()
        log.debug(f"Form data received: {form}")

        caller_phone = form.get("From", "")
        called_number = form.get("To", "")
        call_sid = form.get("CallSid", "")

        log.info(f"Incoming call | from={caller_phone} to={called_number} sid={call_sid}")

        client_config = await lookup_client(caller_phone, called_number, call_sid)

        if not client_config:
            log.warning("No client config - returning fallback TwiML")
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
        log.info(f"Using WS host: {ws_host}")

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

    except Exception as e:
        log.exception("CRASH in /incoming-call")
        return HTMLResponse(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Sorry, something went wrong. Please try again.</Say><Hangup/></Response>',
            media_type="text/xml",
            status_code=500
        )

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
        log.debug(f"Initial WS message: {msg}")

        if msg.get("event") == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid = msg["start"].get("customParameters", {}).get("callSid")
            log.info(f"Stream started | sid={call_sid} | stream={stream_sid}")

            session = session_store.get(call_sid, {})
            log.debug(f"Session data: {session}")

            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=session.get("systemPrompt", "You are a helpful AI receptionist.")
            )

            chat = model.start_chat(history=[])

            # Send initial greeting
            greeting = "Hi, this is Ava with Sunlight Solar. To get started, are you the home owner?"
            response = chat.send_message(greeting)
            log.info(f"Gemini text response: {response.text}")

            # Convert to speech and send
            await send_text_as_audio(ws, stream_sid, greeting)

            async def keep_alive():
                while ws.client_state == "CONNECTED":
                    await asyncio.sleep(5)
                    await ws.send_text(json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": base64.b64encode(b"").decode("utf-8")},
                    }))
                    log.debug("Sent keep-alive ping")

            asyncio.create_task(keep_alive())

            async def twilio_to_gemini():
                log.info("Starting twilio_to_gemini loop")
                try:
                    async for message in _websocket_stream(ws):
                        if message.get("event") == "media":
                            log.info("Received media packet from caller")
                            payload = base64.b64decode(message["media"]["payload"])
                            pcm_8k = audioop.ulaw2lin(payload, 2)
                            pcm_24k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)
                            # Send audio to Gemini (new SDK audio input)
                            response = chat.send_message(Blob(data=pcm_24k, mime_type="audio/pcm;rate=24000"))
                            log.info(f"Gemini audio response received")
                            await send_text_as_audio(ws, stream_sid, response.text)
                        elif message.get("event") == "stop":
                            log.info("Twilio stop event")
                            break
                except Exception as e:
                    log.error(f"twilio_to_gemini error: {e}")

            await twilio_to_gemini()

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected | sid={call_sid}")
    except Exception as e:
        log.exception(f"WebSocket error: {e}")
    finally:
        if call_sid:
            await trigger_post_call(call_sid)

async def send_text_as_audio(ws: WebSocket, stream_sid: str, text: str):
    try:
        log.info(f"Generating TTS for: {text[:50]}...")
        tts = gTTS(text=text, lang="en")
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        audio = AudioSegment.from_mp3(mp3_fp)
        audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
        raw_pcm = audio.raw_data

        mulaw = audioop.lin2ulaw(raw_pcm, 2)

        payload = base64.b64encode(mulaw).decode("utf-8")
        await ws.send_text(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        }))
        log.info(f"Sent TTS audio packet (length {len(mulaw)} bytes)")
    except Exception as e:
        log.error(f"TTS send error: {e}")

# Your helper functions remain the same (lookup_client, handle_booking, trigger_post_call, _websocket_stream)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
