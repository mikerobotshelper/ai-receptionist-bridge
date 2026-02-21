# app.py
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

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("twilio-gemini-bridge")

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-exp"  # or "gemini-1.5-flash-live" when stable

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Optional: load dynamic prompt from n8n or env
SYSTEM_PROMPT = """
You are Ava, a friendly receptionist for Sunlight Solar.
Greet the caller warmly and ask if they are the homeowner.
Speak naturally and clearly. Keep responses short.
"""

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")

    log.info(f"Incoming call - SID: {call_sid}")

    # Return TwiML to start bidirectional media stream
    host = request.headers.get("host", "localhost")
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
    await ws.accept()
    log.info("WebSocket connected")

    call_sid = None
    stream_sid = None

    try:
        # Wait for Twilio 'start' message
        initial_msg = await ws.receive_text()
        msg = json.loads(initial_msg)

        if msg.get("event") == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid = msg["start"].get("customParameters", {}).get("callSid")
            log.info(f"Stream started | call_sid={call_sid} | stream_sid={stream_sid}")

            config = {
                "response_modalities": ["AUDIO"],
                "system_instruction": SYSTEM_PROMPT,
                "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}},
            }

            async with gemini_client.aio.live.connect(model=GEMINI_MODEL, config=config) as gemini_session:
                log.info("Gemini Live session started")

                async def twilio_to_gemini():
                    try:
                        async for message in _websocket_stream(ws):
                            if message.get("event") == "media":
                                payload = base64.b64decode(message["media"]["payload"])
                                # μ-law (8kHz) → linear PCM (16-bit) → resample to 24kHz
                                pcm_8k = audioop.ulaw2lin(payload, 2)
                                pcm_24k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)
                                await gemini_session.send(
                                    input=types.Blob(data=pcm_24k, mime_type="audio/pcm;rate=24000")
                                )
                                log.debug("Sent audio chunk to Gemini")
                            elif message.get("event") == "stop":
                                log.info("Twilio sent stop event")
                                break
                    except Exception as e:
                        log.error(f"Twilio → Gemini error: {e}")

                async def gemini_to_twilio():
                    try:
                        async for response in gemini_session.receive():
                            if response.data:
                                # Gemini returns 24kHz PCM → convert back to 8kHz μ-law for Twilio
                                pcm_8k, _ = audioop.ratecv(response.data, 2, 1, 24000, 8000, None)
                                mulaw = audioop.lin2ulaw(pcm_8k, 2)
                                await ws.send_text(json.dumps({
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": base64.b64encode(mulaw).decode("utf-8")},
                                }))
                                log.debug("Sent audio chunk to Twilio")
                            if response.tool_call:
                                log.warning("Tool call received - not handled yet")
                    except Exception as e:
                        log.error(f"Gemini → Twilio error: {e}")

                await asyncio.gather(twilio_to_gemini(), gemini_to_twilio())

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected | call_sid={call_sid}")
    finally:
        log.info("WebSocket closed")

async def _websocket_stream(ws: WebSocket):
    while True:
        try:
            yield json.loads(await ws.receive_text())
        except:
            break

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
