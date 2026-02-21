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
log = logging.getLogger("receptionist")

app = FastAPI()

GEMINI_API_KEY           = os.environ.get("GEMINI_API_KEY", "")
N8N_CALL_START_URL       = os.environ.get("N8N_CALL_START_URL", "")
N8N_BOOK_APPOINTMENT_URL = os.environ.get("N8N_BOOK_APPOINTMENT_URL", "")
N8N_POST_CALL_URL        = os.environ.get("N8N_POST_CALL_URL", "")

GEMINI_MODEL  = "gemini-2.5-flash-native-audio-latest"
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

session_store: dict[str, dict] = {}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/incoming-call")
async def incoming_call(request: Request):
    try:
        form          = await request.form()
        caller_phone  = form.get("From", "")
        called_number = form.get("To", "")
        call_sid      = form.get("CallSid", "")

        log.info(f"Incoming call | from={caller_phone} to={called_number} sid={call_sid}")

        client_config = await lookup_client(caller_phone, called_number, call_sid)

        if not client_config:
            log.error(f"No client config for {called_number}")
            return HTMLResponse(
                content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Sorry, this number is not configured. Goodbye.</Say><Hangup/></Response>',
                media_type="text/xml",
            )

        session_store[call_sid] = {
            "callerPhone":       caller_phone,
            "calledNumber":      called_number,
            "companyName":       client_config.get("companyName", ""),
            "calendarId":        client_config.get("calendarId", ""),
            "timezone":          client_config.get("timezone", "America/New_York"),
            "systemPrompt":      client_config.get("systemPrompt", "You are a helpful receptionist."),
            "clientRecordId":    client_config.get("clientRecordId", ""),
            "callerName":        "",
            "callerEmail":       "",
            "appointmentBooked": False,
            "appointmentTime":   "",
            "reason":            "",
            "callSummary":       "",
        }

        host  = request.headers.get("host", "")
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Connect>"
            '<Stream url="wss://' + host + '/ws">'
            '<Parameter name="callSid" value="' + call_sid + '"/>'
            "</Stream></Connect></Response>"
        )

        log.info(f"TwiML sent | wss://{host}/ws | sid={call_sid}")
        return HTMLResponse(content=twiml, media_type="text/xml")

    except Exception:
        log.exception("Crash in /incoming-call")
        return HTMLResponse(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Something went wrong. Please try again.</Say><Hangup/></Response>',
            media_type="text/xml",
        )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket accepted")

    call_sid   = None
    stream_sid = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            log.info(f"WS event: {msg.get('event')}")

            if msg.get("event") == "start":
                stream_sid = msg["start"]["streamSid"]
                call_sid   = (
                    msg["start"].get("customParameters", {}).get("callSid")
                    or msg["start"].get("callSid", "")
                )
                log.info(f"Stream started | sid={call_sid} | stream={stream_sid}")
                break
            elif msg.get("event") == "stop":
                log.info("Stop before start — closing")
                return

        if not call_sid or call_sid not in session_store:
            log.error(f"No session for sid={call_sid}")
            return

        session       = session_store[call_sid]
        system_prompt = session.get("systemPrompt", "You are a helpful receptionist.")
        company_name  = session.get("companyName", "our business")

        greeting = (
            "Greet the caller warmly as a receptionist for "
            + company_name
            + ". Use the name and personality from your system prompt."
        )

        gemini_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=system_prompt,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Puck"
                    )
                )
            ),
            tools=[make_booking_tool()],
        )

        async with gemini_client.aio.live.connect(
            model=GEMINI_MODEL, config=gemini_config
        ) as gemini_session:

            # Send greeting as plain string — not types.Part.from_text()
            await gemini_session.send(input=greeting, end_of_turn=True)
            log.info("Greeting sent to Gemini")

            async def twilio_to_gemini():
                async for msg in twilio_stream(ws):
                    if msg.get("event") == "media":
                        mulaw    = base64.b64decode(msg["media"]["payload"])
                        pcm8     = audioop.ulaw2lin(mulaw, 2)
                        pcm16, _ = audioop.ratecv(pcm8, 2, 1, 8000, 16000, None)
                        await gemini_session.send(
                            input=types.Blob(data=pcm16, mime_type="audio/pcm;rate=16000")
                        )
                    elif msg.get("event") == "stop":
                        log.info(f"Twilio stop | sid={call_sid}")
                        break

            async def gemini_to_twilio():
                async for response in gemini_session.receive():
                    if response.data:
                        pcm8, _ = audioop.ratecv(response.data, 2, 1, 24000, 8000, None)
                        mulaw   = audioop.lin2ulaw(pcm8, 2)
                        payload = base64.b64encode(mulaw).decode("utf-8")
                        await ws.send_text(json.dumps({
                            "event":     "media",
                            "streamSid": stream_sid,
                            "media":     {"payload": payload},
                        }))

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
    except Exception:
        log.exception(f"WebSocket error | sid={call_sid}")
    finally:
        if call_sid and call_sid in session_store:
            await trigger_post_call(call_sid)


async def twilio_stream(ws: WebSocket):
    while True:
        try:
            raw = await ws.receive_text()
            yield json.loads(raw)
        except Exception:
            break


def make_booking_tool():
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="book_appointment",
                description=(
                    "Book a calendar appointment. "
                    "Collect callerName, callerEmail, date (YYYY-MM-DD), "
                    "time (HH:MM 24h), and reason BEFORE calling this."
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
                            description="Email address of the caller",
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
                            description="Reason for the appointment",
                        ),
                        "durationMinutes": types.Schema(
                            type=types.Type.INTEGER,
                            description="Duration in minutes, default 60",
                        ),
                    },
                    required=["callerName", "callerEmail", "date", "time", "reason"],
                ),
            )
        ]
    )


async def lookup_client(caller_phone, called_number, call_sid):
    payload = {
        "callerPhone":  caller_phone,
        "calledNumber": called_number,
        "callSid":      call_sid,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(N8N_CALL_START_URL, json=payload)
            log.info(f"Flow A | status={resp.status_code} | body={resp.text[:400]}")
            if resp.status_code == 200 and resp.text.strip():
                data = resp.json()
                if data.get("success"):
                    return data
                log.warning(f"Flow A success=false: {data}")
    except Exception:
        log.exception("lookup_client error")
    return None


async def handle_booking(args, call_sid):
    session = session_store.get(call_sid, {})
    payload = {
        "callerName":      args.get("callerName", ""),
        "callerEmail":     args.get("callerEmail", ""),
        "callerPhone":     session.get("callerPhone", ""),
        "date":            args.get("date", ""),
        "time":            args.get("time", ""),
        "reason":          args.get("reason", ""),
        "durationMinutes": args.get("durationMinutes", 60),
        "calendarId":      session.get("calendarId", ""),
        "timezone":        session.get("timezone", "America/New_York"),
        "companyName":     session.get("companyName", ""),
        "clientRecordId":  session.get("clientRecordId", ""),
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp   = await client.post(N8N_BOOK_APPOINTMENT_URL, json=payload)
            result = resp.json()
            if result.get("success"):
                session_store[call_sid].update({
                    "appointmentBooked": True,
                    "appointmentTime":   args.get("date", "") + " " + args.get("time", ""),
                    "callerName":        args.get("callerName", ""),
                    "callerEmail":       args.get("callerEmail", ""),
                    "reason":            args.get("reason", ""),
                })
                log.info(f"Booking confirmed | sid={call_sid}")
            return result
    except Exception:
        log.exception(f"handle_booking error | sid={call_sid}")
        return {"success": False, "message": "Booking system temporarily unavailable."}


async def trigger_post_call(call_sid):
    data    = session_store.pop(call_sid, {})
    payload = {
        "callerName":        data.get("callerName", "Valued Customer"),
        "callerEmail":       data.get("callerEmail", ""),
        "callerPhone":       data.get("callerPhone", ""),
        "companyName":       data.get("companyName", ""),
        "appointmentTime":   data.get("appointmentTime", ""),
        "reason":            data.get("reason", ""),
        "callSummary":       data.get("callSummary", "Call completed via AI Voice Receptionist."),
        "clientTwilioPhone": data.get("calledNumber", ""),
        "appointmentBooked": data.get("appointmentBooked", False),
    }
    log.info(f"Flow C trigger | sid={call_sid} | booked={payload['appointmentBooked']}")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(N8N_POST_CALL_URL, json=payload)
            log.info(f"Flow C response | {resp.status_code} | {resp.text[:300]}")
    except Exception:
        log.exception(f"trigger_post_call error | sid={call_sid}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
