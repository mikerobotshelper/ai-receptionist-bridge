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

logging.basicConfig(level=logging.DEBUG)  # DEBUG level for more detail
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

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

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
        log.debug(f"Initial WS message: {msg}")

        if msg.get("event") == "start":
            stream_sid = msg["start"]["streamSid"]
            call_sid = msg["start"].get("customParameters", {}).get("callSid")
            log.info(f"Stream started | sid={call_sid} | stream={stream_sid}")

            session = session_store.get(call_sid, {})
            log.debug(f"Session data: {session}")

            config = {
                "response_modalities": ["AUDIO"],
                "system_instruction": session.get(
                    "systemPrompt",
                    "You are Ava, a friendly receptionist for Sunlight Solar. Always speak immediately and clearly. Start every conversation with a greeting and ask if the caller is the homeowner."
                ),
                "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}},
                # "tools": [_get_tools_config()],  # commented to avoid startup issues
            }

            async with gemini_client.aio.live.connect(model=GEMINI_MODEL, config=config) as gemini_session:
                log.info("Gemini Live session connected")

                # Send initial greeting
                greeting = "Hi, this is Ava with Sunlight Solar. To get started, are you the home owner?"
                await gemini_session.send(input=types.Part.from_text(greeting))
                log.info(f"Sent initial greeting: {greeting}")

                # Send follow-up if no response (force audio flow for testing)
                async def send_follow_up():
                    await asyncio.sleep(6)
                    if ws.client_state == "CONNECTED":
                        follow_up = "I didn't hear anything. Are you still there? Please say 'yes' or 'no'."
                        await gemini_session.send(input=types.Part.from_text(follow_up))
                        log.info(f"Sent follow-up: {follow_up}")

                asyncio.create_task(send_follow_up())

                async def twilio_to_gemini():
                    log.info("Starting twilio_to_gemini loop")
                    try:
                        async for message in _websocket_stream(ws):
                            log.debug(f"Twilio event: {message.get('event')}")
                            if message.get("event") == "media":
                                log.info("Received media packet from caller")
                                payload = base64.b64decode(message["media"]["payload"])
                                pcm_8k = audioop.ulaw2lin(payload, 2)
                                pcm_24k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)
                                await gemini_session.send(input=types.Blob(data=pcm_24k, mime_type="audio/pcm;rate=24000"))
                            elif message.get("event") == "stop":
                                log.info("Twilio sent stop event")
                                break
                    except Exception as e:
                        log.error(f"twilio_to_gemini error: {e}")

                async def gemini_to_twilio():
                    log.info("Starting gemini_to_twilio loop")
                    chunk_count = 0
                    try:
                        async for response in gemini_session.receive():
                            chunk_count += 1
                            log.debug(f"Gemini chunk #{chunk_count} received")
                            if response.data:
                                log.info(f"Gemini audio chunk size: {len(response.data)} bytes")
                                pcm_8k, _ = audioop.ratecv(response.data, 2, 1, 24000, 8000, None)
                                mulaw = audioop.lin2ulaw(pcm_8k, 2)
                                log.info(f"Converted to mu-law, payload length: {len(mulaw)}")
                                await ws.send_text(json.dumps({
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": base64.b64encode(mulaw).decode("utf-8")},
                                }))
                                log.info("Forwarded audio to Twilio")
                            if response.tool_call:
                                log.info("Gemini tool call detected")
                                for fc in response.tool_call.function_calls:
                                    tool_result = await handle_booking(fc.args, call_sid)
                                    await gemini_session.send(input=types.LiveClientToolResponse(
                                        function_responses=[types.FunctionResponse(
                                            id=fc.id, name=fc.name, response=tool_result
                                        )]
                                    ))
                    except Exception as e:
                        log.error(f"gemini_to_twilio error: {e}")
                    finally:
                        log.info(f"gemini_to_twilio ended after {chunk_count} chunks")

                await asyncio.gather(twilio_to_gemini(), gemini_to_twilio())

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected | sid={call_sid}")
    except Exception as e:
        log.error(f"WebSocket general error: {e}")
    finally:
        if call_sid:
            log.info("Triggering post-call n8n")
            await trigger_post_call(call_sid)

async def lookup_client(from_number, to_number, sid):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(N8N_CALL_START_URL, json={
                "callerPhone": from_number,
                "calledNumber": to_number,
                "callSid": sid
            })
            log.info(f"n8n status: {resp.status_code}")
            log.debug(f"n8n body: {resp.text}")
            return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        log.error(f"lookup_client error: {e}")
        return None

async def handle_booking(args, call_sid):
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
