import asyncio
import contextlib

import pytest

from livekit.agents import llm, utils
from livekit.agents.inference.sts import (
    STS,
    STSSession,
    _decode_error,
    _ResponseGeneration,
)

pytestmark = pytest.mark.unit


def _make_session() -> STSSession:
    model = STS(
        model="openai/gpt-realtime",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
    )
    return model.session()


class _FakeWS:
    """Enough of a socket for teardown: aclose() closes whatever it is holding."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _new_generation() -> _ResponseGeneration:
    return _ResponseGeneration(
        message_ch=utils.aio.Chan(),
        function_ch=utils.aio.Chan(),
        messages={},
        response_id="resp_1",
    )


@pytest.mark.asyncio
async def test_function_call_emitted_only_when_arguments_complete():
    """Function-call arguments only arrive by output_item.done, so the FunctionCall
    must be emitted there (not at output_item.added, where arguments are empty)."""
    session = _make_session()
    session._current_generation = _new_generation()

    session._handle_response_output_item_added(
        {
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "",
            }
        }
    )
    # nothing should be emitted while arguments are still empty
    assert session._current_generation.function_ch.qsize() == 0

    session._handle_response_output_item_done(
        {
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            }
        }
    )

    fc = session._current_generation.function_ch.recv_nowait()
    assert isinstance(fc, llm.FunctionCall)
    assert fc.id == "fc_1"
    assert fc.call_id == "call_1"
    assert fc.name == "get_weather"
    assert fc.arguments == '{"city": "SF"}'
    # exactly one function call emitted
    assert session._current_generation.function_ch.qsize() == 0


@pytest.mark.asyncio
async def test_incomplete_function_call_on_done_is_skipped():
    """A function_call item missing call_id/name should not emit a partial FunctionCall."""
    session = _make_session()
    session._current_generation = _new_generation()

    session._handle_response_output_item_done(
        {
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "",
                "name": "",
                "arguments": "",
            }
        }
    )
    assert session._current_generation.function_ch.qsize() == 0


@pytest.mark.asyncio
async def test_update_chat_ctx_forwards_function_call_output():
    """Tool results reach the server as conversation.item.create, and only once.

    STS manages conversation history server-side, but function_call_output is a
    client->server event the model needs before it can produce a tool reply, so
    update_chat_ctx must forward it (otherwise tool calls hang)."""
    session = _make_session()
    # Mark started+connected so _send queues onto _msg_ch instead of opening a
    # websocket (_send gates lifecycle startup on _started).
    session._started = True
    session._connected = True
    session._ws = object()

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.items.append(llm.FunctionCallOutput(call_id="call_1", output="sunny", is_error=False))

    await session.update_chat_ctx(chat_ctx)

    ev = session._msg_ch.recv_nowait()
    assert ev == {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
    }

    # Idempotent: the same output is not forwarded twice.
    await session.update_chat_ctx(chat_ctx)
    assert session._msg_ch.qsize() == 0


@pytest.mark.asyncio
async def test_input_audio_transcription_completed_emitted():
    """When transcription is enabled the pipeline skips its own STT, so STS must
    surface the user transcript from OpenAI's transcription.completed event."""
    session = _make_session()

    events: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", events.append)

    session._handle_input_audio_transcription_completed(
        {"item_id": "item_1", "transcript": "what's the weather"}
    )

    assert len(events) == 1
    assert events[0].item_id == "item_1"
    assert events[0].transcript == "what's the weather"
    assert events[0].is_final is True


@pytest.mark.asyncio
async def test_input_audio_transcription_delta_accumulates():
    """Interim deltas emit non-final events carrying the full transcript so far,
    and the final .completed clears the per-item accumulator."""
    session = _make_session()

    events: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", events.append)

    session._handle_input_audio_transcription_delta({"item_id": "item_1", "delta": "what's "})
    session._handle_input_audio_transcription_delta({"item_id": "item_1", "delta": "the weather"})

    assert [e.transcript for e in events] == ["what's ", "what's the weather"]
    assert all(e.is_final is False for e in events)
    assert session._input_transcripts["item_1"] == "what's the weather"

    session._handle_input_audio_transcription_completed(
        {"item_id": "item_1", "transcript": "what's the weather?"}
    )

    assert events[-1].is_final is True
    assert events[-1].transcript == "what's the weather?"
    # accumulator is cleared once the turn finalizes
    assert "item_1" not in session._input_transcripts


@pytest.mark.asyncio
async def test_input_audio_transcription_failed_finalizes_the_partial():
    """A failure closes out the partial as final.

    Captioning has already been handed interim text for this item and has no
    other way to learn the turn is over; dropping it leaves the last thing the
    user said on screen as an unfinished partial.
    """
    session = _make_session()

    events: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", events.append)

    session._input_transcripts["item_1"] = "what's th"
    session._handle_input_audio_transcription_failed(
        {"item_id": "item_1", "error": {"message": "boom"}}
    )

    assert len(events) == 1
    assert events[0].transcript == "what's th"
    assert events[0].is_final is True
    assert "item_1" not in session._input_transcripts
    # and it lands in history, so the turn isn't missing from the transcript
    assert session._chat_ctx.items[-1].text_content == "what's th"


@pytest.mark.asyncio
async def test_input_audio_transcription_failed_without_a_partial_is_quiet():
    session = _make_session()

    events: list[llm.InputTranscriptionCompleted] = []
    session.on("input_audio_transcription_completed", events.append)

    session._handle_input_audio_transcription_failed(
        {"item_id": "item_1", "error": {"message": "boom"}}
    )

    assert events == []


@pytest.mark.asyncio
async def test_update_chat_ctx_sends_items_the_provider_does_not_have():
    """Preloaded history and generate_reply(user_input=...) reach the provider.

    The provider owns the live conversation, but it can only answer what it has
    been told: a locally recorded user message that is never sent leaves the model
    replying to nothing.
    """
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")

    await session.update_chat_ctx(chat_ctx)

    ev = session._msg_ch.recv_nowait()
    assert ev["type"] == "conversation.item.create"
    assert ev["item"]["role"] == "user"
    assert ev["item"]["content"] == [{"type": "input_text", "text": "hello"}]
    assert session._msg_ch.qsize() == 0


@pytest.mark.asyncio
async def test_update_chat_ctx_does_not_echo_back_what_the_model_produced():
    """The model's own turns are already in its conversation.

    Re-sending them would duplicate the reply it just gave.
    """
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    session._current_generation = _new_generation()
    session._handle_response_output_item_added({"item": {"id": "msg_1", "type": "message"}})
    session._handle_response_text_delta({"item_id": "msg_1", "delta": "hi there"})
    session._handle_response_output_item_done({"item": {"id": "msg_1", "type": "message"}})

    await session.update_chat_ctx(session.chat_ctx)
    assert session._msg_ch.qsize() == 0


@pytest.mark.asyncio
async def test_update_chat_ctx_is_idempotent():
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")

    await session.update_chat_ctx(chat_ctx)
    await session.update_chat_ctx(chat_ctx)

    assert session._msg_ch.qsize() == 1


@pytest.mark.asyncio
async def test_replay_session_state_resets_and_requeues():
    """On reconnect the session is fresh: in-flight response futures must fail
    (not hang), per-turn state resets, and non-default tool_choice is re-applied
    since it rides on session.update rather than session.create."""
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    session._response_created_futures["evt_1"] = fut
    session._input_transcripts["item_1"] = "partial"
    session._output_transcripts["item_1"] = "partial reply"
    session._tool_choice = "required"

    session._replay_session_state()

    # pending response future is failed rather than left hanging on a dead turn
    assert fut.done()
    assert isinstance(fut.exception(), llm.RealtimeError)
    assert session._response_created_futures == {}
    assert session._input_transcripts == {}
    assert session._output_transcripts == {}

    # non-default tool_choice is replayed on the fresh session
    ev = session._msg_ch.recv_nowait()
    assert ev == {
        "type": "session.update",
        "session": {"type": "realtime", "tool_choice": "required"},
    }
    assert session._msg_ch.qsize() == 0


@pytest.mark.asyncio
async def test_session_failover_replays_conversation():
    """The gateway can move the call to another deployment without our socket
    ever dropping. The replacement has no conversation, so the notice has to
    trigger the same replay a reconnect does — otherwise the agent keeps talking
    to a model that has forgotten the call."""
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="book me a table for four")
    session._chat_ctx = chat_ctx

    reconnected: list[llm.RealtimeSessionReconnectedEvent] = []
    session.on("session_reconnected", reconnected.append)

    session._handle_session_failover(
        {"type": "session.failover", "model": "gpt-realtime-2", "reason": "provider_unavailable"}
    )

    replayed = [session._msg_ch.recv_nowait() for _ in range(session._msg_ch.qsize())]
    assert any(ev["type"] == "conversation.item.create" for ev in replayed), (
        "the conversation must be rebuilt on the replacement provider"
    )
    assert len(reconnected) == 1


@pytest.mark.asyncio
async def test_failover_mid_reply_is_reported():
    """The reply stops halfway. Closing its channels quietly would present that to
    the pipeline as a finished turn, so the caller hears a sentence stop and then
    silence with nothing saying why."""
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()
    session._handle_response_created({"response": {"id": "resp_1"}})

    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    session._handle_session_failover({"type": "session.failover", "reason": "provider_unavailable"})

    assert len(errors) == 1
    assert errors[0].recoverable is True
    assert session._current_generation is None


@pytest.mark.asyncio
async def test_failover_between_turns_is_not_reported():
    """Nothing was interrupted, so there is nothing to tell the app about."""
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    session._handle_session_failover({"type": "session.failover"})

    assert errors == []


@pytest.mark.asyncio
async def test_session_failover_without_context_loss_is_quiet():
    """context_lost is the whole payload of the notice. A handoff that preserves
    the conversation must not re-send it, or the model sees every turn twice."""
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")
    session._chat_ctx = chat_ctx

    session._handle_session_failover({"type": "session.failover", "context_lost": False})

    assert session._msg_ch.qsize() == 0


@pytest.mark.asyncio
async def test_sent_fnc_outputs_survives_replay():
    """Replay re-creates tool outputs from the recorded context, so the sent-set
    must not be cleared or update_chat_ctx would send each output twice."""
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.items.append(llm.FunctionCallOutput(call_id="call_1", output="sunny", is_error=False))
    await session.update_chat_ctx(chat_ctx)
    session._msg_ch.recv_nowait()  # drain the initial forward

    session._replay_session_state()

    assert session._sent_fnc_outputs == {"call_1"}
    replayed = [session._msg_ch.recv_nowait() for _ in range(session._msg_ch.qsize())]
    assert replayed == [
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        }
    ]

    # and the pipeline re-offering the same output does not duplicate it
    await session.update_chat_ctx(chat_ctx)
    assert session._msg_ch.qsize() == 0


@pytest.mark.asyncio
async def test_replay_session_state_default_tool_choice_no_requeue():
    """A default 'auto' tool_choice and no tools produce no replay events."""
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    session._replay_session_state()

    assert session._msg_ch.qsize() == 0


@pytest.mark.asyncio
async def test_conversation_is_recorded_for_replay():
    """The provider owns live history but loses it when the socket is replaced, so
    completed user turns, agent turns and tool calls are mirrored into chat_ctx."""
    session = _make_session()
    session._current_generation = _new_generation()

    session._handle_input_audio_transcription_completed(
        {"item_id": "user_1", "transcript": "what's the weather"}
    )
    session._handle_response_created({"response": {"id": "resp_1"}})
    session._handle_response_output_item_added({"item": {"id": "msg_1", "type": "message"}})
    session._handle_response_text_delta({"item_id": "msg_1", "delta": "it's "})
    session._handle_response_text_delta({"item_id": "msg_1", "delta": "sunny"})
    session._handle_response_output_item_done({"item": {"id": "msg_1", "type": "message"}})

    items = session.chat_ctx.items
    assert [(i.role, i.text_content) for i in items] == [
        ("user", "what's the weather"),
        ("assistant", "it's sunny"),
    ]
    # the per-item accumulator is released once the turn is recorded
    assert session._output_transcripts == {}


@pytest.mark.asyncio
async def test_chat_ctx_returns_a_copy():
    """Callers mutating the returned context must not corrupt what gets replayed."""
    session = _make_session()
    session._record_item(llm.ChatMessage(id="user_1", role="user", content=["hello"]))

    session.chat_ctx.items.clear()

    assert len(session._chat_ctx.items) == 1


@pytest.mark.asyncio
async def test_record_item_is_idempotent():
    """update_chat_ctx re-offers the whole context on every tool call, so items are
    keyed by id rather than appended blindly."""
    session = _make_session()
    item = llm.ChatMessage(id="user_1", role="user", content=["hello"])

    session._record_item(item)
    session._record_item(item)

    assert len(session._chat_ctx.items) == 1


@pytest.mark.asyncio
async def test_replay_session_state_rebuilds_conversation():
    """A reconnect (including the proactive recycle at max_session_duration) must
    re-send the conversation, otherwise the agent re-greets a caller mid-call."""
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    session._record_item(llm.ChatMessage(id="user_1", role="user", content=["hi there"]))
    session._record_item(
        llm.FunctionCall(id="fc_1", call_id="call_1", name="get_weather", arguments="{}")
    )
    session._record_item(llm.FunctionCallOutput(call_id="call_1", output="sunny", is_error=False))
    session._record_item(llm.ChatMessage(id="msg_1", role="assistant", content=["it's sunny"]))
    # no text content: nothing to recreate on the wire
    session._record_item(llm.ChatMessage(id="msg_2", role="assistant", content=[]))

    session._replay_session_state()

    replayed = [session._msg_ch.recv_nowait() for _ in range(session._msg_ch.qsize())]
    assert [ev["item"] for ev in replayed] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi there"}],
        },
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "it's sunny"}],
        },
    ]
    assert all(ev["type"] == "conversation.item.create" for ev in replayed)


def test_user_transcription_on_by_default():
    """Nobody transcribes the user otherwise: the pipeline skips its own STT when a
    realtime model advertises user_transcription, so the default must be on."""
    model = STS(
        model="openai/gpt-realtime",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
    )

    assert model.capabilities.user_transcription is True
    assert model._opts.input_audio_transcription == {"model": "gpt-4o-mini-transcribe"}


def test_user_transcription_opt_out():
    """Passing None disables transcription explicitly, and capabilities follow so
    the pipeline runs its own STT instead."""
    model = STS(
        model="openai/gpt-realtime",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
        input_audio_transcription=None,
    )

    assert model.capabilities.user_transcription is False
    assert model._opts.input_audio_transcription is None


def test_temperature_is_not_forwarded():
    """The GA Realtime session config dropped temperature, and an unknown session
    field rejects the whole session.update (voice, instructions, tools included)."""
    model = STS(
        model="openai/gpt-realtime",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
        temperature=0.8,
    )

    assert not hasattr(model._opts, "temperature")


def test_session_accepts_turn_detection_disabled():
    """AgentActivity always passes this keyword when it opens the session.

    Without it every ordinary AgentSession startup raises TypeError before the
    socket is even opened.
    """
    model = STS(
        model="openai/gpt-realtime",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
    )

    assert model.capabilities.turn_detection is True
    # the pipeline may substitute its own detection because none was asked for
    assert model.capabilities.can_disable_turn_detection is True

    default_td = model._opts.turn_detection
    session = model.session(turn_detection_disabled=True)
    assert session._opts.turn_detection is None
    # the model's own options are untouched: another session still gets VAD
    assert model._opts.turn_detection == default_td
    assert model.session()._opts.turn_detection == default_td


def test_explicit_turn_detection_is_not_overridable():
    """An explicit turn_detection is a decision, not a default for the pipeline to
    replace. Matches the openai realtime plugin."""
    model = STS(
        model="openai/gpt-realtime",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
        turn_detection={"type": "semantic_vad"},
    )

    assert model.capabilities.can_disable_turn_detection is False


def test_bare_realtime_model_names_are_normalized():
    """agent.py accepts `llm="gpt-realtime"`, but the gateway only resolves
    provider-qualified ids, so a bare name has to be qualified here or the session
    fails to open with "model not found"."""
    model = STS(
        model="gpt-realtime",
        api_key="test-key",
        api_secret="test-secret",
        base_url="https://example.livekit.cloud",
    )

    assert model.model == "openai/gpt-realtime"


@pytest.mark.asyncio
async def test_tools_property_reports_the_tools_in_use():
    """Per-turn tool overrides save `tools`, swap in their own, then restore.

    Reporting a stale empty list makes that restore wipe the agent's tools for the
    rest of the call.
    """
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    async def get_weather() -> str:
        return "sunny"

    tool = llm.function_tool(get_weather)
    await session.update_tools([tool])

    assert [t.__name__ for t in session.tools.function_tools.values()] == ["get_weather"]


@pytest.mark.asyncio
async def test_cancelled_generate_reply_cancels_upstream():
    """The provider has not heard about the interruption.

    Left running it produces the reply anyway and speaks it over whatever the
    caller said instead.
    """
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    fut = session.generate_reply()
    [event_id] = session._response_created_futures.keys()
    while session._msg_ch.qsize():
        session._msg_ch.recv_nowait()

    fut.cancel()
    await asyncio.sleep(0)

    assert session._msg_ch.recv_nowait() == {"type": "response.cancel"}
    assert session._response_created_futures == {}

    # a response.created already in flight is discarded, not announced as a
    # fresh unprompted turn
    generations: list[llm.GenerationCreatedEvent] = []
    session.on("generation_created", generations.append)
    session._handle_response_created(
        {"response": {"id": "resp_1", "metadata": {"client_event_id": event_id}}}
    )

    assert generations == []
    assert session._current_generation is None


@pytest.mark.asyncio
async def test_aclose_fails_pending_replies():
    """Otherwise a caller awaiting generate_reply blocks for the full ten-second
    timeout after the session has already gone away."""
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()

    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    session._response_created_futures["evt_1"] = fut

    session._ws = _FakeWS()
    await session.aclose()

    assert fut.done()
    assert isinstance(fut.exception(), llm.RealtimeError)


@pytest.mark.asyncio
async def test_failed_response_is_reported():
    """A rate-limited turn closes its channels like any other, so without an error
    it is indistinguishable from the model choosing to stay silent."""
    session = _make_session()
    session._handle_response_created({"response": {"id": "resp_1"}})

    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    session._handle_response_done(
        {
            "response": {
                "id": "resp_1",
                "status": "failed",
                "status_details": {
                    "error": {"type": "invalid_request_error", "code": "rate_limit_exceeded"}
                },
            }
        }
    )

    assert len(errors) == 1
    assert "rate_limit_exceeded" in str(errors[0].error)
    assert errors[0].recoverable is True
    assert session._current_generation is None


@pytest.mark.asyncio
async def test_completed_and_cancelled_responses_are_not_reported():
    """A cancellation is an interruption the pipeline asked for."""
    session = _make_session()

    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    for status in ("completed", "cancelled"):
        session._handle_response_created({"response": {"id": "resp_1"}})
        session._handle_response_done({"response": {"id": "resp_1", "status": status}})

    assert errors == []


@pytest.mark.asyncio
async def test_truncate_trims_the_record_to_what_was_heard():
    """chat_ctx is what gets replayed onto a replacement provider. Recording the
    whole generated reply would restore words the caller interrupted and never
    heard, and the model would answer as though it had said them."""
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()
    session._current_generation = _new_generation()

    session._handle_response_output_item_added({"item": {"id": "msg_1", "type": "message"}})
    session._handle_response_text_delta(
        {"item_id": "msg_1", "delta": "the capital of France is Paris"}
    )

    session.truncate(
        message_id="msg_1",
        audio_end_ms=800,
        modalities=["audio"],
        audio_transcript="the capital of France",
    )
    session._handle_response_output_item_done({"item": {"id": "msg_1", "type": "message"}})

    assert session.chat_ctx.items[-1].text_content == "the capital of France"


@pytest.mark.asyncio
async def test_truncate_after_the_turn_is_recorded_still_trims():
    """The pipeline truncates when playback stops, which can land either side of
    the output_item.done that records the turn."""
    session = _make_session()
    session._started = True
    session._connected = True
    session._ws = object()
    session._current_generation = _new_generation()

    session._handle_response_output_item_added({"item": {"id": "msg_1", "type": "message"}})
    session._handle_response_text_delta({"item_id": "msg_1", "delta": "one two three"})
    session._handle_response_output_item_done({"item": {"id": "msg_1", "type": "message"}})

    session.truncate(
        message_id="msg_1", audio_end_ms=300, modalities=["audio"], audio_transcript="one two"
    )

    assert session.chat_ctx.items[-1].text_content == "one two"


@pytest.mark.asyncio
async def test_requests_made_while_the_socket_is_down_are_not_lost():
    """A reply started during a reconnect used to be dropped by the send pump and
    then failed, which reads to the caller as the agent going silent for a turn."""
    session = _make_session()
    session._started = True
    session._connected = False
    session._ws = None

    session._queue_event({"type": "response.create", "response": {}})
    session._queue_event({"type": "input_audio_buffer.append", "audio": "…"})
    send_task = asyncio.create_task(session._send_loop())
    await asyncio.sleep(0)

    # held, not sent: there is no socket to send them on
    assert session._msg_ch.qsize() == 0

    session._connected = True
    session._ws = object()
    session._replay_session_state()

    replayed = [session._msg_ch.recv_nowait() for _ in range(session._msg_ch.qsize())]
    assert replayed == [{"type": "response.create", "response": {}}], (
        "the request is re-queued after the replay; real-time audio from the gap is not"
    )

    send_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await send_task


@pytest.mark.asyncio
async def test_deferred_requests_land_behind_the_replayed_conversation():
    """A response.create the provider sees before the history it refers to is
    answered against an empty conversation."""
    session = _make_session()
    session._started = True
    session._connected = False
    session._ws = None
    session._record_item(llm.ChatMessage(id="user_1", role="user", content=["book a table"]))

    session._queue_event({"type": "response.create", "response": {}})
    send_task = asyncio.create_task(session._send_loop())
    await asyncio.sleep(0)

    session._connected = True
    session._ws = object()
    session._replay_session_state()

    replayed = [session._msg_ch.recv_nowait() for _ in range(session._msg_ch.qsize())]
    assert [ev["type"] for ev in replayed] == [
        "conversation.item.create",
        "response.create",
    ]

    send_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await send_task


def test_error_frames_decode_in_both_shapes():
    """Provider errors arrive nested as OpenAI sends them; the ones the gateway
    raises itself are flat. Reading only the nested form reduced every gateway
    error to the repr of the whole frame and lost the code."""
    assert _decode_error(
        {"type": "error", "code": "model_not_found", "message": "no such model"}
    ) == ("no such model", "model_not_found")
    assert _decode_error(
        {"type": "error", "error": {"code": "insufficient_quota", "message": "out of credit"}}
    ) == ("out of credit", "insufficient_quota")
    # type stands in for a missing code, as the openai plugin does
    assert _decode_error({"error": {"type": "invalid_request_error", "message": "bad"}}) == (
        "bad",
        "invalid_request_error",
    )
    message, code = _decode_error({"type": "error"})
    assert code == ""
    assert message  # never empty: falls back to the frame itself


@pytest.mark.asyncio
async def test_fatal_error_codes_are_not_recoverable():
    """A quota or auth failure fails identically on every later turn, so the
    session is not worth keeping alive."""
    session = _make_session()

    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    session._handle_event(
        {"type": "error", "code": "insufficient_quota", "message": "out of credit"}
    )
    session._handle_event({"type": "error", "code": "server_error", "message": "try again"})

    assert [e.recoverable for e in errors] == [False, True]


@pytest.mark.asyncio
async def test_generation_done_gates_session_recycle():
    """The recycle timer waits on _generation_done, which clears while a turn is
    in flight and sets once the generation closes, so a proactive reconnect lands
    between turns rather than mid-response."""
    session = _make_session()
    assert session._generation_done.is_set()  # idle at start

    session._handle_response_created({"response": {"id": "resp_1"}})
    assert not session._generation_done.is_set()  # turn in flight, recycle held off

    session._close_current_generation()
    assert session._generation_done.is_set()  # turn done, recycle may proceed
