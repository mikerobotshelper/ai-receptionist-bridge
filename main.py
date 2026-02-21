import json
import logging
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("receptionist")

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
N8N_CALL_START_URL = os.environ.get("N8N_CALL_START_URL")
N8N_BOOK_APPOINTMENT_URL = os.environ.get("N8N_BOOK_APPOINTMENT_URL")
N8N_POST_CALL_URL = os.environ.get("N8N_POST_CALL_URL")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# In-memory store per call
call_store: dict = {}


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
            return twiml_response("<Say>Sorry, this number is not configured. Goodbye.</Say><Hangup/>")

        company_name = client_config.get("companyName", "our business")
        system_prompt = client_config.get("systemPrompt", "You are a helpful receptionist.")

        call_store[call_sid] = {
            "callerPhone": caller_phone,
            "calledNumber": called_number,
            "companyName": company_name,
            "calendarId": client_config.get("calendarId", ""),
            "timezone": client_config.get("timezone", "UTC"),
            "systemPrompt": system_prompt,
            "clientRecordId": client_config.get("clientRecordId", ""),
            "history": [],
            "appointmentBooked": False,
            "callerName": "",
            "callerEmail": "",
            "appointmentTime": "",
            "reason": "",
        }

        greeting = await ask_gemini(
            call_sid,
            f"Greet the caller warmly as the receptionist for {company_name}. Be friendly and ask how you can help. Keep it to 1-2 sentences.",
        )
        log.info(f"Greeting: {greeting}")

        host = request.headers.get("host", "")
        gather_url = f"https://{host}/gather"

        return twiml_response(
            f'<Say voice="Polly.Joanna">{escape_xml(greeting)}</Say>'
            f'<Gather input="speech" action="{gather_url}" method="POST" '
            f'speechTimeout="2" speechModel="phone_call" enhanced="true" timeout="10">'
            f"</Gather>"
            f'<Say voice="Polly.Joanna">I didn\'t catch that. Please call back and try again.</Say>'
            f"<Hangup/>"
        )

    except Exception:
        log.exception("CRASH in /incoming-call")
        return twiml_response("<Say>Sorry, something went wrong. Please try again.</Say><Hangup/>")


# ──────────────────────────────────────────────
# STEP 2 — Twilio sends caller speech here after each turn
# ──────────────────────────────────────────────
@app.post("/gather")
async def gather(request: Request):
    try:
        form = await request.form()
        call_sid = form.get("CallSid", "")
        speech_result = form.get("SpeechResult", "").strip()
        log.info(f"Speech | sid={call_sid} | text='{speech_result}'")

        host = request.headers.get("host", "")
        gather_url = f"https://{host}/gather"

        if not speech_result:
            return twiml_response(
                '<Say voice="Polly.Joanna">Sorry, I didn\'t catch that. Could you say that again?</Say>'
                f'<Gather input="speech" action="{gather_url}" method="POST" '
                f'speechTimeout="2" speechModel="phone_call" enhanced="true" timeout="10">'
                f"</Gather>"
                "<Hangup/>"
            )

        session = call_store.get(call_sid)
        if not session:
            return twiml_response("<Say>Sorry, your session expired. Please call back.</Say><Hangup/>")

        # Check if caller wants to end the call
        lower = speech_result.lower()
        if any(w in lower for w in ["goodbye", "bye", "hang up", "that's all", "thank you goodbye"]):
            farewell = await ask_gemini(call_sid, speech_result)
            await trigger_post_call(call_sid)
            return twiml_response(
                f'<Say voice="Polly.Joanna">{escape_xml(farewell)}</Say><Hangup/>'
            )

        gemini_reply = await ask_gemini(call_sid, speech_result)
        log.info(f"Gemini reply: {gemini_reply}")

        # Check if Gemini wants to book an appointment
        if "[[BOOK_APPOINTMENT:" in gemini_reply:
            return await process_booking(call_sid, gemini_reply, gather_url)

        return twiml_response(
            f'<Say voice="Polly.Joanna">{escape_xml(gemini_reply)}</Say>'
            f'<Gather input="speech" action="{gather_url}" method="POST" '
            f'speechTimeout="2" speechModel="phone_call" enhanced="true" timeout="10">'
            f"</Gather>"
            f'<Say voice="Polly.Joanna">Are you still there?</Say>'
            f'<Gather input="speech" action="{gather_url}" method="POST" '
            f'speechTimeout="2" speechModel="phone_call" enhanced="true" timeout="10">'
            f"</Gather>"
            f"<Hangup/>"
        )

    except Exception:
        log.exception("CRASH in /gather")
        return twiml_response("<Say>Sorry, something went wrong. Please try again.</Say><Hangup/>")


# ──────────────────────────────────────────────
# GEMINI — Text chat with conversation history
# ──────────────────────────────────────────────
async def ask_gemini(call_sid: str, user_message: str) -> str:
    session = call_store.get(call_sid, {})
    system_prompt = session.get("systemPrompt", "You are a helpful receptionist.")
    history = session.get("history", [])
    calendar_id = session.get("calendarId", "")
    timezone = session.get("timezone", "UTC")

    booking_instruction = (
        "\n\nWhen the caller wants to book an appointment and you have their name, email, "
        "preferred date, time, and reason — include this marker at the very end of your reply "
        "(after your spoken words):\n"
        '[[BOOK_APPOINTMENT:{"callerName":"NAME","callerEmail":"EMAIL",'
        '"date":"YYYY-MM-DD","time":"HH:MM","reason":"REASON","durationMinutes":60}]]\n'
        f"Calendar ID: {calendar_id}\nTimezone: {timezone}"
    )

    full_system = system_prompt + booking_instruction

    # Build messages list for Gemini
    messages = []
    for h in history:
        role = "user" if h["role"] == "user" else "model"
        messages.append(types.Content(role=role, parts=[types.Part(text=h["parts"][0])]))

    messages.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=full_system),
            contents=messages,
        )
        reply = response.text.strip()

        # Save to history
        history.append({"role": "user", "parts": [user_message]})
        history.append({"role": "model", "parts": [reply]})
        session["history"] = history

        return reply

    except Exception:
        log.exception(f"ask_gemini error | sid={call_sid}")
        return "I'm sorry, I had a technical issue. Could you repeat that?"


# ──────────────────────────────────────────────
# BOOKING — Parse Gemini marker and call Flow B
# ──────────────────────────────────────────────
async def process_booking(call_sid: str, gemini_reply: str, gather_url: str) -> HTMLResponse:
    session = call_store.get(call_sid, {})

    try:
        parts = gemini_reply.split("[[BOOK_APPOINTMENT:")
        spoken_part = parts[0].strip()
        booking_json_str = parts[1].rstrip("]]").strip()
        booking_data = json.loads(booking_json_str)

        booking_data["calendarId"] = session.get("calendarId")
        booking_data["timezone"] = session.get("timezone")
        booking_data["companyName"] = session.get("companyName")
        booking_data["clientRecordId"] = session.get("clientRecordId")
        booking_data["callerPhone"] = session.get("callerPhone")

        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.post(N8N_BOOK_APPOINTMENT_URL, json=booking_data)
        result = resp.json()

        if result.get("success"):
            session["appointmentBooked"] = True
            session["appointmentTime"] = booking_data.get("date", "") + " " + booking_data.get("time", "")
            session["callerName"] = booking_data.get("callerName", "")
            session["callerEmail"] = booking_data.get("callerEmail", "")
            session["reason"] = booking_data.get("reason", "")
            log.info(f"Booking confirmed | sid={call_sid}")

            confirmation = spoken_part or "Your appointment is confirmed! You'll receive a confirmation email shortly."
            return twiml_response(
                f'<Say voice="Polly.Joanna">{escape_xml(confirmation)}</Say>'
                f'<Gather input="speech" action="{gather_url}" method="POST" '
                f'speechTimeout="2" speechModel="phone_call" enhanced="true" timeout="10">'
                f"</Gather>"
                f"<Hangup/>"
            )
        else:
            sorry = "I'm sorry, that time slot isn't available. Would you like to try a different time?"
            return twiml_response(
                f'<Say voice="Polly.Joanna">{escape_xml(sorry)}</Say>'
                f'<Gather input="speech" action="{gather_url}" method="POST" '
                f'speechTimeout="2" speechModel="phone_call" enhanced="true" timeout="10">'
                f"</Gather>"
                f"<Hangup/>"
            )

    except Exception:
        log.exception(f"process_booking error | sid={call_sid}")
        sorry = "I had trouble booking that. Could you try again?"
        return twiml_response(
            f'<Say voice="Polly.Joanna">{escape_xml(sorry)}</Say>'
            f'<Gather input="speech" action="{gather_url}" method="POST" '
            f'speechTimeout="2" speechModel="phone_call" enhanced="true" timeout="10">'
            f"</Gather>"
            f"<Hangup/>"
        )


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
            return None

        data = json.loads(body)
        return data if data.get("success") else None

    except Exception:
        log.exception("lookup_client failed")
        return None


# ──────────────────────────────────────────────
# POST-CALL — Trigger n8n Flow C after hang up
# ──────────────────────────────────────────────
async def trigger_post_call(call_sid: str):
    data = call_store.pop(call_sid, {})
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
            "callSummary": "Call completed via voice receptionist.",
            "clientTwilioPhone": data.get("calledNumber", ""),
            "clientRecordId": data.get("clientRecordId", ""),
            "appointmentBooked": data.get("appointmentBooked", False),
        }
        log.info(f"Flow C trigger | sid={call_sid} | booked={payload['appointmentBooked']}")
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.post(N8N_POST_CALL_URL, json=payload)
        log.info(f"Flow C response | {resp.status_code}")
    except Exception:
        log.exception(f"trigger_post_call error | sid={call_sid}")


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def twiml_response(body: str) -> HTMLResponse:
    return HTMLResponse(
        content=f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>',
        media_type="text/xml",
    )


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
