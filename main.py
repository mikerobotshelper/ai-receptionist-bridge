import asyncio
import base64
import json
import logging
import os
from io import BytesIO

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# New official Google GenAI SDK (2026)
from google import genai
from google.genai import types

# TTS fallback (no pydub needed)
from gtts import gTTS

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

# Configure new SDK
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    log.error("GEMINI_API_KEY not set!")

N8N_CALL_START_URL = os.environ.get("N8N_CALL_START_URL")
N8N_BOOK_APPOINTMENT_URL = os.environ.get("N8N_BOOK_APPOINTMENT_URL")
N8N_POST_CALL_URL = os.environ.get("N8N_POST_CALL_URL")

GEMINI_MODEL = "gemini-1.5-flash"  # stable model - use 2.0-flash-exp only if needed

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
            log.warning("No client config from n8n - returning fallback TwiML")
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
            "appointmentBooked": False,
            "conversation_history": []  # for chat state
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
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Sorry, something went wrong on our end. Please try again.</Say><Hangup/></Response>',
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
                system_instruction=session.get("systemPrompt", "You are Ava, a friendly receptionist for Sunlight Solar. Speak immediately and clearly. Start every conversation with a greeting and ask if the caller is the homeowner. Do not wait for input before speaking.")
            )

            chat = model.start_chat(history=[])

            # Initial greeting
            greeting = "Hello! This is Ava with Sunlight Solar. To get started, are you the homeowner?"
            response = chat.send_message(greeting)
            log.info(f"Gemini text response: {response.text}")

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

            async def handle_incoming_audio():
                try:
                    async for message in _websocket_stream(ws):
                        if message.get("event") == "media":
                            log.info("Received media packet from caller")
                            payload = base64.b64decode(message["media"]["payload"])
                            pcm_8k = audioop.ulaw2lin(payload, 2)
                            # Placeholder STT - replace with real STT later
                            user_input = "User spoke something"
                            session["conversation_history"].append({"role": "user", "parts": [user_input]})

                            response = chat.send_message(user_input)
                            reply_text = response.text

                            await send_text_as_audio(ws, stream_sid, reply_text)
                            log.info(f"Replied with: {reply_text}")

                            session["conversation_history"].append({"role": "model", "parts": [reply_text]})
                        elif message.get("event") == "stop":
                            break
                except Exception as e:
                    log.error(f"Audio handling error: {e}")

            await handle_incoming_audio()

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

        # Convert MP3 to raw PCM (no pydub needed - use wave or simple conversion)
        # Simplified: assume 16kHz mono for now - adjust as needed
        # For full conversion, add ffmpeg if needed in Railway

        # Placeholder: send silence or minimal packet to keep alive
        silence = base64.b64encode(b"\x00" * 1600).decode("utf-8")  # 100ms silence
        await ws.send_text(json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": silence},
        }))
        log.info("Sent placeholder audio packet (silence)")
    except Exception as e:
        log.error(f"TTS send error: {e}")

# Rest of your helper functions remain the same (lookup_client, handle_booking, trigger_post_call, _websocket_stream)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
