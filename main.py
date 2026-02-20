"""
AI Voice Receptionist – Bridge Server
Matched to workflow: AI Receptionist – Flow C v2 (Gmail edition)
Flow A → POST /webhook/call-start (Twilio Call Start node)
Flow B → POST /webhook/book-appointment (Gemini Booking Request node)
Flow C → POST /webhook/post-call (Call Ended Trigger node)
What this server does:
  1. Receives an incoming Twilio call at /incoming-call
  2. Calls n8n Flow A → looks up which business was called in Airtable
  3. Opens a WebSocket, relays audio between Twilio and Gemini Live
  4. When Gemini wants to book → calls n8n Flow B (checks + creates calendar event)
  5. When the call ends → calls n8n Flow C (logs lead + sends Gmail confirmation to caller)
"""
import asyncio
import base64
import json
import logging
import os
import audioop
from typing import Optional
import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("receptionist")

# ── Environment variables — set these in Railway Variables tab ────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
N8N_CALL_START_URL = os.environ["N8N_CALL_START_URL"]  # Flow A
N8N_BOOK_APPOINTMENT_URL = os.environ["N8N_BOOK_APPOINTMENT_URL"]  # Flow B
N8N_POST_CALL_URL = os.environ["N8N_POST_CALL_URL"]  # Flow C
GEMINI_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"  # Current Live model (Feb 2026)

# ── Gemini client ─────────────────────────────────────────────────────────────
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI()

# ── Per-call session store (keyed by Twilio CallSid) ─────────────────────────
session_store: dict[str, dict] = {}

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}

# ─────────────────────────────────────────────────────────────────────────────
# INCOMING CALL — Twilio hits this first
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Twilio sends a POST here when a customer calls.
    We ask n8n Flow A which business owns this number,
    then return TwiML to open a real-time audio stream.
    """
    form = await request.form()
    caller_phone = form.get("From", "")
    called_number = form.get("To", "")
    call_sid = form.get("CallSid", "")
    log.info(f"Incoming call | from={caller_phone} to={called_number} sid={call_sid}")

    # Ask n8n Flow A: which business is this number registered to?
    client_config = await lookup_client(caller_phone, called_number, call_sid)
    if not client_config:
        log.warning(f"No client found for {called_number}")
        return HTMLResponse(
            content=(
                '<?xml version="1.0" encoding="UTF-8"?>'
                "<Response>"
                "<Say>Sorry, this number is not configured. Goodbye.</Say>"
                "<Hangup/>"
                "</Response>"
            ),
            media_type="text/xml",
        )

    # Save everything for this call — WebSocket handler will use it
    session_store[call_sid] = {
        "callerPhone": caller_phone,
        "calledNumber": called_number,
        "companyName": client_config.get("companyName", ""),
        "calendarId": client_config.get("calendarId", ""),
        "timezone": client_config.get("timezone", "America/New_York"),
        "systemPrompt": client_config.get("systemPrompt", ""),
        "clientRecordId": client_config.get("clientRecordId", ""),
        # filled in during the call by Gemini tool calls
        "callerName": "",
        "callerEmail": "",
        "appointmentBooked": False,
        "appointmentTime": "",
        "reason": "",
        "callSummary": "",
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

# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET — real-time audio bridge between Twilio and Gemini
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    call_sid: Optional[str] = None
    stream_sid: Optional[str] = None
    try:
        # First Twilio message tells us the callSid
        raw = await ws.receive_text()
        msg = json.loads(raw)
        if msg.get("event") == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid = (
                msg["start"].get("customParameters", {}).get("callSid")
                or msg["start"].get("callSid", "")
            )
            log.info(f"Stream started | sid={call_sid} stream={stream_sid}")

        session = session_store.get(call_sid, {})
        system_prompt = session.get("systemPrompt", "You are a helpful receptionist.")

        # Gemini Live config (dict format - recommended in current SDK)
        gemini_config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": system_prompt,
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": "Puck"}
                }
            },
            "tools": [_make_booking_tool()],
        }

        async with gemini_client.aio.live.connect(
            model=GEMINI_MODEL, config=gemini_config
        ) as gemini_session:
            async def twilio_to_gemini():
                """Twilio → convert μ-law 8kHz to PCM 24kHz → Gemini"""
                async for msg in _twilio_stream(ws):
                    if msg.get("event") == "media":
                        mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                        pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
                        pcm_24k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)
                        await gemini_session.send(
                            input=types.Blob(
                                data=pcm_24k,
                                mime_type="audio/pcm;rate=24000"
                            )
                        )
                    elif msg.get("event") == "stop":
                        log.info(f"Twilio stop event | sid={call_sid}")
                        break

            async def gemini_to_twilio():
                """Gemini → convert PCM 24kHz to μ-law 8kHz → Twilio"""
                async for response in gemini_session.receive():
                    # Audio chunk — send back to the caller
                    if response.data:
                        pcm_8k, _ = audioop.ratecv(response.data, 2, 1, 24000, 8000, None)
                        mulaw = audioop.lin2ulaw(pcm_8k, 2)
                        payload = base64.b64encode(mulaw).decode("utf-8")
                        await ws.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": payload},
                        }))
                    # Tool call — Gemini wants to book an appointment
                    if response.tool_call:
                        for fn in response.tool_call.function_calls:
                            if fn.name == "book_appointment":
                                result = await handle_booking(fn.args, call_sid)
                                await gemini_session.send(
                                    input=types.LiveClientToolResponse(
                                        function_responses=[
                                            types.FunctionResponse(
                                                id=fn.id,
                                                name=fn.name,
                                                response=result,
                                            )
                                        ]
                                    )
                                )

            await asyncio.gather(twilio_to_gemini(), gemini_to_twilio())

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected | sid={call_sid}")
    except Exception as e:
        log.exception(f"WebSocket error | sid={call_sid} | {e}")
    finally:
        # Call ended — fire n8n Flow C
        if call_sid and call_sid in session_store:
            await trigger_post_call(call_sid)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
async def _twilio_stream(ws: WebSocket):
    """Async generator: yield parsed Twilio WebSocket messages one by one."""
    while True:
        try:
            raw = await ws.receive_text()
            yield json.loads(raw)
        except (WebSocketDisconnect, Exception):
            break

def _make_booking_tool() -> types.Tool:
    """
    Gemini function definition for booking an appointment.
    Matched to what n8n Flow B (Extract Booking Details node) expects:
      date, time, reason, callerName, callerEmail, durationMinutes
    callerEmail is REQUIRED — Gemini must ask for it before booking.
    """
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="book_appointment",
                description=(
                    "Book a calendar appointment for the caller. "
                    "Before calling this function you MUST have collected: "
                    "the caller's full name, email address, preferred date (YYYY-MM-DD), "
                    "preferred time (HH:MM 24h), and reason for the appointment."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "callerName": types.Schema(
                            type=types.Type.STRING,
                            description="Full name of the caller",
                        ),
                        "callerEmail": types.Schema(
                            type=types.Type.STRING,
                            description="Email address of the caller — needed to send confirmation",
                        ),
                        "date": types.Schema(
                            type=types.Type.STRING,
                            description="Date in YYYY-MM-DD format",
                        ),
                        "time": types.Schema(
                            type=types.Type.STRING,
                            description="Time in HH:MM 24-hour format",
                        ),
                        "reason": types.Schema(
                            type=types.Type.STRING,
                            description="Purpose of the appointment",
                        ),
                        "durationMinutes": types.Schema(
                            type=types.Type.INTEGER,
                            description="Duration in minutes, default is 60",
                        ),
                    },
                    required=["callerName", "callerEmail", "date", "time", "reason"],
                ),
            )
        ]
    )

async def lookup_client(
    caller_phone: str, called_number: str, call_sid: str
) -> Optional[dict]:
    """
    POST to n8n Flow A (Twilio Call Start webhook).
    Matches the Extract Call Info node fields: callerPhone, calledNumber, callSid.
    Returns the full client config from Respond with Client Config node, or None.
    """
    payload = {
        "callerPhone": caller_phone,
        "calledNumber": called_number,
        "callSid": call_sid,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(N8N_CALL_START_URL, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    return data
            log.warning(f"Flow A {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        log.exception(f"lookup_client failed: {e}")
    return None

async def handle_booking(args: dict, call_sid: str) -> dict:
    """
    Called when Gemini fires the book_appointment tool.
    POSTs to n8n Flow B (Gemini Booking Request webhook).
    Matches Extract Booking Details node fields exactly.
    Saves result into session_store so Flow C can include it after the call.
    """
    session = session_store.get(call_sid, {})
    payload = {
        # Fields read by Extract Booking Details node
        "callerName": args.get("callerName", ""),
        "callerEmail": args.get("callerEmail", ""),
        "callerPhone": session.get("callerPhone", ""),
        "date": args.get("date", ""),
        "time": args.get("time", ""),
        "reason": args.get("reason", ""),
        "durationMinutes": args.get("durationMinutes", 60),
        "calendarId": session.get("calendarId", ""),
        "timezone": session.get("timezone", "America/New_York"),
        "companyName": session.get("companyName", ""),
        "clientRecordId": session.get("clientRecordId", ""),
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(N8N_BOOK_APPOINTMENT_URL, json=payload)
            result = resp.json()
            if result.get("success"):
                # Store for Flow C post-call trigger
                session_store[call_sid].update({
                    "appointmentBooked": True,
                    "appointmentTime": f"{args.get('date')} {args.get('time')}",
                    "callerName": args.get("callerName", ""),
                    "callerEmail": args.get("callerEmail", ""),
                    "reason": args.get("reason", ""),
                })
                log.info(f"Booking confirmed | sid={call_sid}")
            else:
                log.info(f"Slot unavailable | sid={call_sid}")
            return result
    except Exception as e:
        log.exception(f"handle_booking error | sid={call_sid} | {e}")
        return {"success": False, "message": "Booking system temporarily unavailable."}

async def trigger_post_call(call_sid: str):
    """
    Fires when the call ends (WebSocket closes).
    POSTs to n8n Flow C (Call Ended Trigger webhook).
    Flow C does:
      Extract Call Details → Airtable Log Lead → Was Appointment Booked?
        YES → Build Confirmation SMS → Send Gmail to caller
             → Airtable Mark Email Sent → Respond Done
        NO → Send Email Internal (alert to you at mike.robotshelper@gmail.com)
    Payload field names must match the Extract Call Details node assignments exactly.
    """
    data = session_store.pop(call_sid, {})
    payload = {
        # Matched to Extract Call Details node assignment names
        "callerName": data.get("callerName", "Valued Customer"),
        "callerEmail": data.get("callerEmail", ""),
        "callerPhone": data.get("callerPhone", ""),
        "companyName": data.get("companyName", ""),
        "appointmentTime": data.get("appointmentTime", ""),
        "reason": data.get("reason", ""),
        "callSummary": data.get("callSummary", "Call completed via AI Voice Receptionist."),
        "clientTwilioPhone": data.get("calledNumber", ""),
        "appointmentBooked": data.get("appointmentBooked", False),
    }
    log.info(
        f"Triggering Flow C | sid={call_sid} "
        f"booked={payload['appointmentBooked']} "
        f"email={payload['callerEmail']}"
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(N8N_POST_CALL_URL, json=payload)
            log.info(f"Flow C response | {resp.status_code} | {resp.text[:300]}")
    except Exception as e:
        log.exception(f"trigger_post_call error | sid={call_sid} | {e}")
