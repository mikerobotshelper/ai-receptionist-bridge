import asyncio
import base64
import json
import logging
import os
from typing import Optional

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Correct import for Gemini SDK
import google.generativeai as genai

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

GEMINI_MODEL = "gemini-1.5-flash"  # Use stable model for reliability

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
            log.warning("No client config from n8n - returning fallback")
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
        log.info(f"Using WebSocket host: {ws_host}")

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
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Sorry, something went wrong. Please try again later.</Say><Hangup/></Response>',
            media_type="text/xml",
            status_code=500
        )

# The rest of your WebSocket and helper functions go here...
# (websocket_endpoint, lookup_client, handle_booking, trigger_post_call, _websocket_stream, _get_tools_config)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
