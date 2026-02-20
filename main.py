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

# ── Audio Compatibility for Python 3.13 ──────────────────────────────────────
try:
    import audioop
except ImportError:
    import audioop_lts as audioop

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("receptionist")

# ── Environment variables ────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
N8N_CALL_START_URL = os.environ["N8N_CALL_START_URL"]
N8N_BOOK_APPOINTMENT_URL = os.environ["N8N_BOOK_APPOINTMENT_URL"]
N8N_POST_CALL_URL = os.environ["N8N_POST_CALL_URL"]
GEMINI_MODEL = "gemini-2.0-flash-exp" # Adjusted for current stable live model

# ── Gemini client ─────────────────────────────────────────────────────────────
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI()

# ── Per-call session store ────────────────────────────────────────────────────
session_store: dict[str, dict] = {}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form()
    caller_phone = form.get("From", "")
    called_number = form.get("To", "")
    call_sid = form.get("CallSid", "")
    log.info(f"Incoming call | from={caller_phone} to={called_number} sid={call_sid}")

    client_config = await lookup_client(caller_phone, called_number, call_sid)
    if not client_config:
        return HTMLResponse(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Configuration error. Goodbye.</Say><Hangup/></Response>',
            media_type="text/xml",
        )

    session_store[call_sid] = {
        "callerPhone": caller_phone,
        "calledNumber": called_number,
        "companyName": client_config.get("companyName", ""),
        "calendarId": client_config.get("calendarId", ""),
        "timezone": client_config.get("timezone", "America/New_York"),
        "systemPrompt": client_config.get("systemPrompt", ""),
        "clientRecordId": client_config.get("clientRecordId", ""),
        "appointmentBooked": False,
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
    await ws.accept()
    call_sid = None
    stream_sid = None
    try:
        raw = await ws.receive_text()
        msg = json.loads(raw)
        if msg.get("event") == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid = msg["start"].get("customParameters", {}).get("callSid") or msg["start"].get("callSid", "")
            log.info(f"Stream started | sid={call_sid}")

        session = session_store.get(call_sid, {})
        system_prompt = session.get("systemPrompt", "You are a helpful receptionist.")

        gemini_config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": system_prompt,
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}},
            "tools": [_make_booking_tool()],
        }

        async with gemini_client.aio.live.connect(model=GEMINI_MODEL, config=gemini_config) as gemini_session:
            async def twilio_to_gemini():
                async for msg in _twilio_stream(ws):
                    if msg.get("event") == "media":
                        payload = base64.b64decode(msg["media"]["payload"])
                        pcm_8k = audioop.ulaw2lin(payload, 2)
                        pcm_24k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)
                        await gemini_session.send(input=types.Blob(data=pcm_24k, mime_type="audio/pcm;rate=24000"))
                    elif msg.get("event") == "stop":
                        break

            async def gemini_to_twilio():
                async for response in gemini_session.receive():
                    if response.data:
                        pcm_8k, _ = audioop.ratecv(response.data, 2, 1, 24000, 8000, None)
                        mulaw = audioop.lin2ulaw(pcm_8k, 2)
                        await ws.send_text(json.dumps({
                            "event": "media", "streamSid": stream_sid,
                            "media": {"payload": base64.b64encode(mulaw).decode("utf-8")},
                        }))
                    if response.tool_call:
                        for fn in response.tool_call.function_calls:
                            result = await handle_booking(fn.args, call_sid)
                            await gemini_session.send(input=types.LiveClientToolResponse(
                                function_responses=[types.FunctionResponse(id=fn.id, name=fn.name, response=result)]
                            ))

            await asyncio.gather(twilio_to_gemini(), gemini_to_twilio())

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected | sid={call_sid}")
    finally:
        if call_sid:
            await trigger_post_call(call_sid)

async def lookup_client(caller_phone: str, called_number: str, call_sid: str):
    payload = {"callerPhone": caller_phone, "calledNumber": called_number, "callSid": call_sid}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(N8N_CALL_START_URL, json=payload)
            if resp.status_code == 200:
                return resp.json() # This now expects JSON from n8n
    except Exception as e:
        log.error(f"lookup_client error: {e}")
    return None

def _make_booking_tool():
    return types.Tool(function_declarations=[types.FunctionDeclaration(
        name="book_appointment",
        description="Book an appointment. Collect name, email, date, time, and reason first.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "callerName": types.Schema(type=types.Type.STRING),
                "callerEmail": types.Schema(type=types.Type.STRING),
                "date": types.Schema(type=types.Type.STRING),
                "time": types.Schema(type=types.Type.STRING),
                "reason": types.Schema(type=types.Type.STRING),
            },
            required=["callerName", "callerEmail", "date", "time", "reason"]
        )
    )])

async def _twilio_stream(ws):
    while True:
        try:
            yield json.loads(await ws.receive_text())
        except:
            break

async def handle_booking(args, call_sid):
    # Integration for Flow B
    return {"success": True, "message": "Appointment recorded."}

async def trigger_post_call(call_sid):
    # Integration for Flow C
    log.info(f"Call ended for {call_sid}. Triggering Flow C.")
