import json
import logging
import os

import httpx
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse

import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("receptionist")

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
N8N_CALL_START_URL = os.environ.get("N8N_CALL_START_URL")
N8N_BOOK_APPOINTMENT_URL = os.environ.get("N8N_BOOK_APPOINTMENT_URL")
N8N_POST_CALL_URL = os.environ.get("N8N_POST_CALL_URL")

genai.configure(api_key=GEMINI_API_KEY)

# In-memory conversation store per call
# Stores: system_prompt, history, company data
call_store: dict = {}


# ──────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────
# STEP 1 — Twilio calls this when phone rings
# Returns TwiML that greets the caller and starts listening
# ──────────────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    try:
        form = await request.form()
        caller_phone = form.get("From", "")
        called_number = form.get("To", "")
        call_sid = form.get("CallSid", "")
        log.info(f"Incoming call | from={caller_phone} to={called_number} sid={call_sid}")

        # Look up which business this number belongs to
        client_config = await lookup_client(caller_phone, called_number, call_sid)

        if not client_config:
            log.warning(f"No client config for {called_number}")
            return twiml_response(
                "<Say>Sorry, this number is not configured. Goodbye.</Say><Hangup/>"
            )

        company_name = client_config.get("companyName", "our business")
        system_prompt = client_config.get("systemPrompt", "You are a helpful receptionist.")

        # Store session data for this call
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

        # Generate greeting using Gemini
        greeting = await ask_gemini(
            call_sid,
            f"Greet the caller warmly as the receptionist for {company_name}. Be friendly and ask how you can help. Keep it brief — 1-2 sentences.",
        )

        log.info(f"Greeting: {greeting}")

        # Return TwiML: say the greeting, then listen for caller's response
        host = request.headers.get("host", "")
        gather_url = f"https://{host}/gather"

        return twiml_response(
            f'<Say voice="Polly.Joanna">{escape_xml(greeting)}</Say>'
            f'<Gather input="speech" action="{gather_url}" method="POST" '
            f'speechTimeout="2" speechModel="phone_call" enhanced="true" timeout="10">'
            f'<Say voice="Polly.Joanna">I\'m listening.</Say>'
            f"</Gather>"
            f'<Say voice="Polly.Joanna">I didn\'t catch that. Please call back and try again.</Say>'
            f"<Hangup/>"
        )

    except Exception:
        log.exception("CRASH in /incoming-call")
        return twiml_response("<Say>Sorry, something went wrong. Please try again.</Say><Hangup/>")


# ──────────────────────────────────────────────
# STEP 2 — Twilio sends the caller's speech here
# Gemini processes it and we respond with more TwiML
# ──────────────────────────────────────────────
@app.post("/gather")
async def gather(request: Request):
    try:
        form = await request.form()
        call_sid = form.get("CallSid", "")
        speech_result = form.get("SpeechResult", "").strip()
        confidence = form.get("Confidence", "0")

        log.info(f"Speech | sid={call_sid} | text='{speech_result}' | confidence={confidence}")

        if not speech_result:
            return twiml_response(
                '<Say voice="Polly.Joanna">Sorry, I didn\'t catch that. Could you say that again?</Say>'
                f'<Gather input="speech" action="https://{request.headers.get("host")}/gather" '
                f'method="POST" speechTimeout="2" speechModel="phone_call" enhanced="true" timeout="10">'
                f"</Gather>"
                "<Hangup/>"
            )

        session = call_store.get(call_sid)
        if not session:
            return twiml_response(
                "<Say>Sorry, your session expired. Please call back.</Say><Hangup/>"
            )

        # Check if caller wants to end the call
        lower = speech_result.lower()
        if any(word in lower for word in ["goodbye", "bye", "hang up", "end call", "that's all", "thank you goodbye"]):
            farewell = await ask_gemini(call_sid, speech_result)
            await trigger_post_call(call_sid)
            return twiml_response(
                f'<Say voice="Polly.Joanna">{escape_xml(farewell)}</Say><Hangup/>'
            )

        # Check if this is a booking request — let Gemini decide
        gemini_reply = await ask_gemini(call_sid, speech_result)

        # Check if Gemini wants to book an appointment
        # (Gemini will include a special marker if booking is needed)
        if "[[BOOK_APPOINTMENT:" in gemini_reply:
            booking_result = await process_booking(call_sid, gemini_reply, request)
            return booking_result

        log.info(f"Gemini reply: {gemini_reply}")

        host = request.headers.get("host", "")
        gather_url = f"https://{host}/gather"

        # Speak Gemini's reply and listen for the next caller input
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
        return twiml_response(
            "<Say>Sorry, something went wrong. Please try again.</Say><Hangup/>"
        )


# ──────────────────────────────────────────────
# GEMINI — Ask Gemini and maintain conversation history
# ──────────────────────────────────────────────
async def ask_gemini(call_sid: str, user_message: str) -> str:
    session = call_store.get(call_sid, {})
    system_prompt = session.get("systemPrompt", "You are a helpful receptionist.")
    history = session.get("history", [])
    company_name = session.get("companyName", "our business")
    calendar_id = session.get("calendarId", "")
    timezone = session.get("timezone", "UTC")

    # Add booking instruction to system prompt
    booking_instruction = (
        "\n\nIMPORTANT: When the caller wants to book an appointment and you have collected their "
        "name, email, preferred date, time, and reason — respond with ONLY this exact format on the "
        "last line of your response (after your spoken reply):\n"
        "[[BOOK_APPOINTMENT:{\"callerName\":\"NAME\",\"callerEmail\":\"EMAIL\","
        "\"date\":\"YYYY-MM-DD\",\"time\":\"HH:MM\",\"reason\":\"REASON\",\"durationMinutes\":60}]]\n"
        f"Calendar ID: {calendar_id}\nTimezone: {timezone}"
    )

    full_system = system_prompt + booking_instruction

    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=full_system,
        )

        chat = model.start_chat(history=history)
        response = chat.send_message(user_message)
        reply = response.text.strip()

        # Update conversation history
        history.append({"role": "user", "parts": [user_message]})
        history.append({"role": "model", "parts": [reply]})
        session["history"] = history

        return reply

    except Exception:
        log.exception(f"ask_gemini error | sid={call_sid}")
        return "I'm sorry, I had a technical issue. Could you repeat that?"


# ──────────────────────────────────────────────
# BOOKING — Parse Gemini's booking marker and call Flow B
# ──────────────────────────────────────────────
async def process_booking(call_sid: str, gemini_reply: str, request: Request) -> HTMLResponse:
    session = call_store.get(call_sid, {})
    host = request.headers.get("host", "")
    gather_url = f"https://{host}/gather"

    try:
        # Split reply from booking marker
        parts = gemini_reply.split("[[BOOK_APPOINTMENT:")
        spoken_part = parts[0].strip()
        booking_json_str = parts[1].rstrip("]]").strip()
        booking_data = json.loads(booking_json_str)

        # Add session context
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

            confirmation = spoken_part or "Your appointment has been booked! You'll receive a confirmation email shortly."
            log.info(f"Booking confirmed | sid={call_sid}")

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
        if not data.get("success"):
            return None

        return data

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
