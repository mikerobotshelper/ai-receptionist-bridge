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
                "system_instruction": "You are Ava, a friendly receptionist for Sunlight Solar. Speak immediately and clearly. Start every conversation with a greeting and ask if the caller is the homeowner. Do not wait for input before speaking.",
                "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}},
                # "tools": [_get_tools_config()],  # still commented
            }

            async with gemini_client.aio.live.connect(model=GEMINI_MODEL, config=config) as gemini_session:
                log.info("Gemini Live session connected")

                # Send strong initial prompt to force speech
                greeting = "Hello! This is Ava with Sunlight Solar. I'm excited to help you with solar options. Are you the homeowner? Please say yes or no."
                await gemini_session.send(input=types.Part.from_text(greeting))
                log.info(f"Sent initial greeting: {greeting}")

                # Force a second message after 4 seconds if no user input
                async def send_follow_up():
                    await asyncio.sleep(4)
                    if ws.client_state == "CONNECTED":
                        follow_up = "I didn't hear a response. Are you still there? Let's get started — are you the homeowner?"
                        await gemini_session.send(input=types.Part.from_text(follow_up))
                        log.info(f"Sent follow-up: {follow_up}")

                asyncio.create_task(send_follow_up())

                async def twilio_to_gemini():
                    log.info("Starting twilio_to_gemini loop")
                    try:
                        async for message in _websocket_stream(ws):
                            log.debug(f"Twilio event: {message}")
                            if message.get("event") == "media":
                                log.info("Received media packet from caller - processing")
                                payload = base64.b64decode(message["media"]["payload"])
                                pcm_8k = audioop.ulaw2lin(payload, 2)
                                pcm_24k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 24000, None)
                                await gemini_session.send(input=types.Blob(data=pcm_24k, mime_type="audio/pcm;rate=24000"))
                                log.info("Sent caller audio to Gemini")
                            elif message.get("event") == "stop":
                                log.info("Twilio stop event")
                                break
                    except Exception as e:
                        log.error(f"twilio_to_gemini error: {e}")

                async def gemini_to_twilio():
                    log.info("Starting gemini_to_twilio loop - waiting for Gemini to speak")
                    chunk_count = 0
                    try:
                        async for response in gemini_session.receive():
                            chunk_count += 1
                            log.info(f"Gemini chunk #{chunk_count} received")
                            if response.data:
                                log.info(f"Gemini audio chunk size: {len(response.data)} bytes")
                                pcm_8k, _ = audioop.ratecv(response.data, 2, 1, 24000, 8000, None)
                                mulaw = audioop.lin2ulaw(pcm_8k, 2)
                                log.info(f"Converted mu-law payload length: {len(mulaw)}")
                                await ws.send_text(json.dumps({
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": base64.b64encode(mulaw).decode("utf-8")},
                                }))
                                log.info("Forwarded audio packet to Twilio")
                            else:
                                log.debug(f"Gemini chunk #{chunk_count} has no data")
                            if response.tool_call:
                                log.info("Gemini tool call detected")
                                for fc in response.tool_call.function_calls:
                                    log.info(f"Tool call: {fc.name} with args {fc.args}")
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
