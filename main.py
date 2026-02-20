# ... (imports unchanged)

GEMINI_MODEL = "gemini-2.5-flash-native-audio"  # ← Updated to current model (Feb 2026)

# ... (rest of env vars same)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ... (app, session_store, /health, /incoming-call mostly unchanged)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    call_sid: Optional[str] = None
    stream_sid: Optional[str] = None
    try:
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

        # Updated config as dict (more stable in recent SDK)
        gemini_config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": system_prompt,
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": "Puck"}
                }
            },
            "tools": [_make_booking_tool().to_dict() if hasattr(_make_booking_tool(), 'to_dict') else _make_booking_tool()],
        }

        async with gemini_client.aio.live.connect(
            model=GEMINI_MODEL, config=gemini_config
        ) as gemini_session:
            async def twilio_to_gemini():
                async for msg in _twilio_stream(ws):
                    if msg.get("event") == "media":
                        mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                        pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
                        # Try 16kHz first; change to 24000 if audio quality/compatibility issues
                        pcm_converted, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
                        await gemini_session.send(
                            input=types.Blob(
                                data=pcm_converted,
                                mime_type="audio/pcm;rate=16000"
                            )
                        )
                    elif msg.get("event") == "stop":
                        log.info(f"Twilio stop event | sid={call_sid}")
                        break

            async def gemini_to_twilio():
                async for response in gemini_session.receive():
                    if response.data:
                        pcm_8k, _ = audioop.ratecv(response.data, 2, 1, 24000, 8000, None)
                        mulaw = audioop.lin2ulaw(pcm_8k, 2)
                        payload = base64.b64encode(mulaw).decode("utf-8")
                        await ws.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": payload},
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
    except Exception as e:
        log.exception(f"WebSocket error | sid={call_sid} | {e}")
    finally:
        if call_sid and call_sid in session_store:
            await trigger_post_call(call_sid)

# ... (helpers unchanged, except add .to_dict() fallback if needed in tool)

def _make_booking_tool() -> types.Tool:
    tool = types.Tool(...)  # your existing
    return tool
