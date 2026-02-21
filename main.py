import asyncio
import base64
import json
import logging
import os

import httpx
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse

from google import genai
from google.genai import types

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("receptionist")

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
N8N_CALL_START_URL = os.environ.get("N8N_CALL_START_URL")
N8N_BOOK_APPOINTMENT_URL = os.environ.get("N8N_BOOK_APPOINTMENT_URL")
N8N_POST_CALL_URL = os.environ.get("N8N_POST_CALL_URL")
WEBSOCKET_HOST = os.environ.get("WEBSOCKET_HOST", "")

# This model supports true bidirectional native audio
GEMINI_MODEL = "gemini-2.0-flash-live-001"

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={"api_version": "v1alpha"},
)

session_store: dict = {}


# ──────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────
# STEP 1 — Twilio calls this when phone rings
# ──────────────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    try:
        form = await request.form()
        caller_phone = form.get("From", "")
        called_number = form.get("To", "")
        call_sid = form.get("CallSid", "")
        log.info(f"Incoming call | from={caller_phone} to={called_number} sid={call_sid}")

        client_config = await lookup_client(caller_phone, called_number, call_sid)

        if not client_config:
            log.warning(f"No client config for {called_number}")
            return HTMLResponse(
                content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Sorry, this number is not configured. Goodbye.</Say><Hangup/></Response>',
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
            "callerName": "",
            "callerEmail": "",
            "appointmentTime": "",
            "reason": "",
            "callSummary": "",
        }

        ws_host = WEBSOCKET_HOST or request.headers.get("host", "localhost")
        log.info(f"TwiML sent | wss://{ws_host}/ws | sid={call_sid}")

        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Connect>"
            '<Stream url="wss://' + ws_host + '/ws">'
            '<Parameter name="callSid" value="' + call_sid + '"/>'
            "</Stream></Connect></Response>"
        )
        return HTMLResponse(content=twiml, media_type="text/xml")

    except Exception:
        log.exception("CRASH in /incoming-call")
        return HTMLResponse(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Sorry, something went wrong. Please try again later.</Say><Hangup/></Response>',
            media_type="text/xml",
            status_code=500,
        )


# ──────────────────────────────────────────────
# STEP 2 — WebSocket bridges Twilio ↔ Gemini
# ──────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket accepted")

    call_sid = None
    stream_sid = None

    try:
        # Wait for Twilio "start" event (skip "connected" first)
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")
            log.info(f"WS event: {event}")

            if event == "connected":
                continue
            if event == "start":
                call_sid = msg["start"]["customParameters"]["callSid"]
                stream_sid = msg["start"]["streamSid"]
                log.info(f"Stream started | sid={call_sid} | stream={stream_sid}")
                break
            if event == "stop":
                return

        data = session_store.get(call_sid, {})
        if not data:
            log.error(f"No session data for {call_sid}")
            return

        system_prompt = data.get("systemPrompt", "You are a helpful receptionist.")
        company_name = data.get("companyName", "our business")

        gemini_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=system_prompt,
            tools=[_get_tools_config()],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Aoede"
                    )
                )
            ),
        )

        async with gemini_client.aio.live.connect(
            model=GEMINI_MODEL, config=gemini_config
        ) as gemini_session:

            log.info(f"Gemini session open | sid={call_sid}")

            # Send initial greeting — Gemini speaks first
            greeting = (
                "Greet the caller warmly as the receptionist for "
                + company_name
                + ". Be friendly and ask how you can help."
            )
            await gemini_session.send(input=greeting, end_of_turn=True)
            log.info(f"Greeting sent | sid={call_sid}")

            # Run both audio directions simultaneously
            await asyncio.gather(
                _twilio_to_gemini(ws, gemini_session, call_sid),
                _gemini_to_twilio(gemini_session, ws, stream_sid, call_sid, data),
            )

    except Exception:
        log.exception(f"WebSocket error | sid={call_sid}")

    finally:
        if call_sid:
            await trigger_post_call(call_sid)
        log.info(f"Connection closed | sid={call_sid}")


# ──────────────────────────────────────────────
# DIRECTION 1 — Caller audio: Twilio → Gemini
# ──────────────────────────────────────────────
async def _twilio_to_gemini(ws: WebSocket, gemini_session, call_sid: str):
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "media":
                # Decode base64 audio from Twilio (mulaw 8kHz)
                audio_bytes = base64.b64decode(msg["media"]["payload"])

                # Convert mulaw → linear PCM 16-bit
                pcm_8k = audioop.ulaw2lin(audio_bytes, 2)

                # Upsample 8kHz → 16kHz for Gemini
                pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)

                # Stream caller audio to Gemini
                await gemini_session.send(
                    input=types.LiveClientRealtimeInput(
                        media_chunks=[
                            types.Blob(
                                data=pcm_16k,
                                mime_type="audio/pcm;rate=16000",
                            )
                        ]
                    )
                )

            elif event == "stop":
                log.info(f"Twilio stop | sid={call_sid}")
                break

    except Exception:
        log.exception(f"_twilio_to_gemini error | sid={call_sid}")


# ──────────────────────────────────────────────
# DIRECTION 2 — Agent audio: Gemini → Twilio
# ──────────────────────────────────────────────
async def _gemini_to_twilio(
    gemini_session, ws: WebSocket, stream_sid: str, call_sid: str, session_data: dict
):
    try:
        async for response in gemini_session.receive():

            # Handle appointment booking function calls
            if response.tool_call:
                for fc in response.tool_call.function_calls:
                    if fc.name == "book_appointment":
                        args = dict(fc.args)
                        result = await handle_booking(args, call_sid, session_data)
                        await gemini_session.send(
                            input=types.LiveClientToolResponse(
                                function_responses=[
                                    types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response=result,
                                    )
                                ]
                            )
                        )

            # Send Gemini audio back to Twilio
            if response.data:
                # Gemini outputs PCM 24kHz — downsample to 8kHz
                pcm_8k, _ = audioop.ratecv(response.data, 2, 1, 24000, 8000, None)

                # Convert linear PCM → mulaw for Twilio
                mulaw = audioop.lin2ulaw(pcm_8k, 2)
                payload = base64.b64encode(mulaw).decode("utf-8")

                await ws.send_text(
                    json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": payload},
                    })
                )

    except Exception:
        log.exception(f"_gemini_to_twilio error | sid={call_sid}")


# ──────────────────────────────────────────────
# LOOKUP — Ask n8n Flow A who is being called
# ──────────────────────────────────────────────
async def lookup_client(caller_phone: str, called_number: str, call_sid: str):
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.post(
                N8N_CALL_START_URL,
                json={
                    "callerPhone": caller_phone,
                    "calledNumber": called_number,
                    "callSid": call_sid,
                },
            )
        log.info(f"Flow A | status={resp.status_code} | body={resp.text[:300]}")

        if resp.status_code != 200:
            return None

        body = resp.text.strip()
        if not body:
            log.error("Flow A returned empty body")
            return None

        data = json.loads(body)
        if not data.get("success"):
            log.warning(f"Flow A success=false: {data}")
            return None

        return data

    except Exception:
        log.exception("lookup_client failed")
        return None


# ──────────────────────────────────────────────
# BOOKING — Send to n8n Flow B → Google Calendar
# ──────────────────────────────────────────────
async def handle_booking(args: dict, call_sid: str, session_data: dict):
    try:
        payload = {
            **args,
            "calendarId": session_data.get("calendarId"),
            "timezone": session_data.get("timezone"),
            "companyName": session_data.get("companyName"),
            "clientRecordId": session_data.get("clientRecordId"),
            "callerPhone": session_data.get("callerPhone"),
        }
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.post(N8N_BOOK_APPOINTMENT_URL, json=payload)
        result = resp.json()

        if result.get("success"):
            session_data["appointmentBooked"] = True
            session_data["appointmentTime"] = args.get("date", "") + " " + args.get("time", "")
            session_data["callerName"] = args.get("callerName", "")
            session_data["callerEmail"] = args.get("callerEmail", "")
            session_data["reason"] = args.get("reason", "")
            log.info(f"Booking confirmed | sid={call_sid}")

        return result

    except Exception:
        log.exception(f"handle_booking error | sid={call_sid}")
        return {"success": False, "error": "Booking failed"}


# ──────────────────────────────────────────────
# POST-CALL — Trigger n8n Flow C after hang up
# ──────────────────────────────────────────────
async def trigger_post_call(call_sid: str):
    data = session_store.pop(call_sid, {})
    if not data:
        return
    try:
        payload = {
            "callerName": data.get("callerName", "Valued Customer"),
            "callerEmail": data.get("callerEmail", ""),
            "callerPhone": data.get("callerPhone", ""),
            "companyName": data.get("companyName", ""),
            "appointmentTime": data.get("appointmentTime", ""),
            "reason": data.get("reason", ""),
            "callSummary": data.get("callSummary", "Call completed."),
            "clientTwilioPhone": data.get("calledNumber", ""),
            "clientRecordId": data.get("clientRecordId", ""),
            "appointmentBooked": data.get("appointmentBooked", False),
        }
        log.info(f"Flow C trigger | sid={call_sid} | booked={payload['appointmentBooked']}")
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.post(N8N_POST_CALL_URL, json=payload)
        log.info(f"Flow C response | {resp.status_code} | {resp.text[:200]}")

    except Exception:
        log.exception(f"trigger_post_call error | sid={call_sid}")


# ──────────────────────────────────────────────
# GEMINI TOOLS — Appointment booking definition
# ──────────────────────────────────────────────
def _get_tools_config():
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="book_appointment",
                description="Book an appointment for the caller in Google Calendar",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "callerName": types.Schema(
                            type=types.Type.STRING,
                            description="Full name of the caller",
                        ),
                        "callerEmail": types.Schema(
                            type=types.Type.STRING,
                            description="Email address of the caller for confirmation",
                        ),
                        "date": types.Schema(
                            type=types.Type.STRING,
                            description="Appointment date in YYYY-MM-DD format",
                        ),
                        "time": types.Schema(
                            type=types.Type.STRING,
                            description="Appointment time in HH:MM 24-hour format",
                        ),
                        "reason": types.Schema(
                            type=types.Type.STRING,
                            description="Reason or purpose of the appointment",
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


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
