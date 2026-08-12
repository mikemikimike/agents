from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents._exceptions import APIConnectionError, APIError, APITimeoutError
from livekit.agents.log import logger
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._utils import create_access_token, get_default_inference_url, get_inference_headers

STSModels = Literal[
    "openai/gpt-realtime",
    "openai/gpt-realtime-mini",
    "openai/gpt-realtime-1.5",
]

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# Reconnection: how many times to retry re-establishing a dropped realtime
# session before surfacing a fatal error, and the base backoff in seconds
# (doubled each attempt) between tries.
_RECONNECT_MAX_RETRIES = 3
_RECONNECT_BASE_BACKOFF = 1.0

# How long to wait for the gateway to accept the WebSocket upgrade.
_CONNECT_TIMEOUT = 30.0

# OpenAI Realtime caps a session at ~30 minutes. Proactively recycle the
# connection before that hard cap so a fresh session is established at a quiet
# point (between turns) instead of the server dropping mid-response. Matches the
# openai realtime plugin's DEFAULT_MAX_SESSION_DURATION.
_DEFAULT_MAX_SESSION_DURATION = 20 * 60

# Transcribe the user's audio unless the caller opts out with
# input_audio_transcription=None. Without it neither side transcribes: the
# realtime session isn't asked to, and the pipeline skips its own STT whenever a
# realtime model advertises user_transcription, so the user's turns would never
# reach the transcript or history. Matches the openai realtime plugin's
# DEFAULT_INPUT_AUDIO_TRANSCRIPTION.
_DEFAULT_INPUT_AUDIO_TRANSCRIPTION: dict[str, Any] = {"model": "gpt-4o-mini-transcribe"}

# Realtime model ids the inference gateway serves, matched on prefix so a new
# point release (gpt-realtime-2.1) needs no change here.
_REALTIME_MODEL_PREFIXES = ("gpt-realtime",)
_REALTIME_MODEL_PROVIDER = "openai"

# How many events may be held while the socket is down. A reconnect takes a few
# seconds at most, so this only has to cover a turn's worth of requests.
_MAX_DEFERRED_EVENTS = 64


def is_realtime_model(model: str) -> bool:
    """Report whether a model string names a speech-to-speech realtime model.

    Used to decide whether ``llm="…"`` resolves to STS or to an ordinary LLM.
    Accepts the bare form as well as the provider-prefixed one, because that is
    what people type.
    """
    _, _, name = model.rpartition("/")
    return name.startswith(_REALTIME_MODEL_PREFIXES)


def _normalize_model(model: str) -> str:
    """Add the provider prefix a bare realtime model id is missing.

    Every other inference modality documents ``provider/model`` and the gateway
    resolves nothing else: a bare "gpt-realtime" tokenizes as the provider
    "gpt-realtime" with no model and fails the session. Accepting the bare form
    and repairing it here is what keeps ``AgentSession(llm="gpt-realtime")``
    working, which is the form the docs use.
    """
    if "/" in model:
        return model
    return f"{_REALTIME_MODEL_PROVIDER}/{model}"


# Codes that will fail identically on the next turn, so a session carrying one is
# not worth keeping alive. Matches the openai realtime plugin's set.
_FATAL_ERROR_CODES = frozenset(
    {
        "insufficient_quota",
        "invalid_api_key",
        "account_deactivated",
        "billing_hard_limit_reached",
    }
)


def _decode_error(data: dict[str, Any]) -> tuple[str, str]:
    """Pull the message and code out of an error frame, in either shape.

    Provider errors arrive nested as OpenAI sends them
    (``{"error": {"message", "code"}}``); the ones the gateway raises itself are
    flat (``{"type": "error", "code", "message"}``). Reading only the nested form
    reduced every gateway error to the repr of the whole frame and lost the code
    that says whether the session can continue.
    """
    nested = data.get("error")
    if isinstance(nested, dict):
        message = nested.get("message") or ""
        code = nested.get("code") or nested.get("type") or ""
        if message or code:
            return message or str(data), str(code)

    message = data.get("message") or ""
    code = data.get("code") or ""
    return (message or str(data)), str(code)


def _build_tool_defs(tools: list[llm.Tool]) -> list[dict[str, Any]]:
    """Convert agent tools to the realtime session's tool schema.

    Shared by update_tools (session-level tools) and generate_reply
    (per-response tools) so the two stay in sync.
    """
    tool_defs: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, llm.FunctionTool):
            tool_defs.append(llm.utils.build_legacy_openai_schema(tool, internally_tagged=True))
        elif isinstance(tool, llm.RawFunctionTool):
            desc = dict(tool.info.raw_schema)
            desc.pop("meta", None)
            desc["type"] = "function"
            tool_defs.append(desc)
    return tool_defs


def _to_realtime_tool_choice(tool_choice: llm.ToolChoice | None) -> Any:
    """Map an llm.ToolChoice to the OpenAI Realtime wire form.

    "auto"/"required"/"none" pass through; a named choice collapses to
    {"type": "function", "name": ...}; None (reset) and any unrecognized value
    fall back to "auto". Mirrors the openai realtime plugin's to_oai_tool_choice
    so STS behaves the same without importing the plugin.
    """
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "function", "name": tool_choice["function"]["name"]}
    return "auto"


def _chat_item_to_realtime_item(item: llm.ChatItem) -> dict[str, Any] | None:
    """Render a chat item as a realtime ``conversation.item.create`` item.

    Returns None for items that can't be recreated on the wire (items with no
    text content, e.g. an assistant turn whose transcript never arrived). Content
    types per role follow the realtime schema: user/system take ``input_text``,
    assistant takes ``output_text``.
    """
    if isinstance(item, llm.FunctionCall):
        return {
            "type": "function_call",
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
        }
    if isinstance(item, llm.FunctionCallOutput):
        return {
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": item.output,
        }
    if not isinstance(item, llm.ChatMessage):
        return None

    # Images and audio are dropped: replaying them would re-upload the original
    # media, and the text transcript is what the conversation depends on.
    text = "".join(c for c in item.content if isinstance(c, str)).strip()
    if not text:
        return None

    role = item.role
    if role == "developer":
        role = "system"
    if role not in ("user", "assistant", "system"):
        return None

    content_type = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": content_type, "text": text}],
    }


@dataclass
class _STSOptions:
    model: str
    voice: str
    instructions: str
    modalities: list[Literal["text", "audio"]]
    base_url: str
    api_key: str
    api_secret: str
    turn_detection: dict[str, Any] | None
    input_audio_transcription: dict[str, Any] | None
    noise_reduction: dict[str, Any] | None
    max_session_duration: float | None


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    messages: dict[str, _MessageGeneration]
    response_id: str
    created_timestamp: float = 0.0
    first_token_timestamp: float | None = None


class STS(llm.RealtimeModel):
    def __init__(
        self,
        model: STSModels | str = "openai/gpt-realtime",
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        instructions: str = "",
        modalities: list[Literal["text", "audio"]] | None = None,
        temperature: float | None = None,  # deprecated, ignored by GA Realtime
        turn_detection: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        noise_reduction: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        td = (
            turn_detection
            if is_given(turn_detection)
            else {"type": "server_vad", "silence_duration_ms": 300}
        )
        transcription = (
            input_audio_transcription
            if is_given(input_audio_transcription)
            else _DEFAULT_INPUT_AUDIO_TRANSCRIPTION
        )
        if temperature is not None:
            logger.warning(
                "STS: temperature is deprecated and ignored; the GA Realtime API "
                "removed it from session configuration"
            )
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=td is not None,
                user_transcription=transcription is not None,
                # OpenAI Realtime (which this proxies) does not auto-reply after a
                # function_call_output; the client must send response.create. False
                # makes the pipeline generate the tool reply explicitly (matches the
                # openai realtime plugin). True would leave tool calls hanging until
                # the pipeline's 5s auto-reply timeout.
                auto_tool_reply_generation=False,
                audio_output="audio" in (modalities or ["text", "audio"]),
                manual_function_calls=False,
                # The pipeline may substitute its own turn detection, but only
                # when the caller didn't ask for a specific one — an explicit
                # turn_detection is a decision, not a default to override.
                # Applied per session (see session), so the model stays reusable.
                can_disable_turn_detection=not is_given(turn_detection),
                # instructions and tools are updatable mid-session via session.update
                # (update_instructions/update_tools), which lets agent handoff patch
                # the live session instead of tearing it down and reconnecting.
                mutable_instructions=True,
                mutable_tools=True,
            )
        )

        lk_base_url = base_url if is_given(base_url) else get_default_inference_url()
        lk_api_key = (
            api_key
            if is_given(api_key)
            else os.getenv("LIVEKIT_INFERENCE_API_KEY", os.getenv("LIVEKIT_API_KEY", ""))
        )
        if not lk_api_key:
            raise ValueError(
                "api_key is required, either as argument or set LIVEKIT_API_KEY env var"
            )

        lk_api_secret = (
            api_secret
            if is_given(api_secret)
            else os.getenv("LIVEKIT_INFERENCE_API_SECRET", os.getenv("LIVEKIT_API_SECRET", ""))
        )
        if not lk_api_secret:
            raise ValueError(
                "api_secret is required, either as argument or set LIVEKIT_API_SECRET env var"
            )

        self._opts = _STSOptions(
            model=_normalize_model(model),
            voice=voice if is_given(voice) else "alloy",
            instructions=instructions,
            modalities=modalities or ["text", "audio"],
            base_url=lk_base_url,
            api_key=lk_api_key,
            api_secret=lk_api_secret,
            turn_detection=td,
            input_audio_transcription=transcription,
            noise_reduction=noise_reduction if is_given(noise_reduction) else None,
            max_session_duration=max_session_duration
            if is_given(max_session_duration)
            else _DEFAULT_MAX_SESSION_DURATION,
        )

    @classmethod
    def from_model_string(cls, model: str) -> STS:
        return cls(model=model)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "livekit"

    def session(self, *, turn_detection_disabled: bool = False) -> STSSession:
        return STSSession(self, turn_detection_disabled=turn_detection_disabled)

    async def aclose(self) -> None:
        pass


class STSSession(llm.RealtimeSession):
    def __init__(self, realtime_model: STS, *, turn_detection_disabled: bool = False) -> None:
        super().__init__(realtime_model)
        self._model: STS = realtime_model
        # Per-session copy: update_instructions and a pipeline-disabled turn
        # detection both change these, and the model is documented as reusable
        # across sessions. Sharing the model's options let one session's
        # instructions leak into the next.
        self._opts = dataclasses.replace(realtime_model._opts)
        if turn_detection_disabled:
            # None serialises as JSON null, which the realtime API reads as
            # manual turns. Omitting the field instead would let the gateway
            # apply its server_vad default and re-enable the very thing the
            # pipeline turned off to run its own turn detection.
            self._opts.turn_detection = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._chat_ctx = llm.ChatContext.empty()
        # ids of the items the provider is known to hold, either because it
        # produced them or because we sent them. Cleared whenever the session
        # moves to a fresh socket, which is what makes the replay re-send
        # everything. Without it there is no way to tell an item the model
        # already has from one it has never been told about.
        self._remote_item_ids: set[str] = set()
        # Tracks the last tool_choice pushed via update_options (stored in the
        # realtime wire form) so we only emit a session.update when it actually
        # changes. "auto" is the realtime default, so a no-op update stays silent.
        self._tool_choice: Any = "auto"
        # call_ids of function_call_output items already forwarded to the server,
        # so update_chat_ctx stays idempotent across repeated calls.
        self._sent_fnc_outputs: set[str] = set()
        # latest tools pushed via update_tools, retained so they can be re-applied
        # after a reconnect (tools ride on session.update, not session.create).
        self._current_tools: list[llm.Tool] = []
        # accumulates streamed user input-audio transcript deltas per item_id so
        # interim events carry the full transcript so far, not just the last chunk.
        self._input_transcripts: dict[str, str] = {}
        # accumulates the agent's own output text per item_id, so a completed
        # assistant turn can be recorded in _chat_ctx for replay after a reconnect.
        self._output_transcripts: dict[str, str] = {}
        # what the caller actually heard of an interrupted assistant turn, keyed
        # by item_id. Overrides the accumulated transcript when the turn is
        # recorded (see truncate).
        self._truncated_transcripts: dict[str, str] = {}
        self._recv_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        # _started: lifecycle tasks have been launched (stays True across
        # reconnects). _connected: a live socket is currently up (False during the
        # reconnect window). Keeping them distinct stops _send from spinning up a
        # duplicate connection while _reconnect is mid-flight.
        self._started = False
        self._connected = False
        self._closing = False

        self._current_generation: _ResponseGeneration | None = None
        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        # client_event_ids whose generate_reply was cancelled before the provider
        # acknowledged it. A response.created already in flight for one of these
        # is dropped instead of being spoken after the interruption.
        self._discarded_event_ids: set[str] = set()
        # events the send pump could not deliver because the socket was down,
        # re-queued after the replay once it is back (see _defer_event).
        self._deferred_events: list[dict[str, Any]] = []
        # Set whenever no generation is in flight. The session-recycle timer waits
        # on this so a proactive reconnect happens between turns, never mid-response.
        self._generation_done = asyncio.Event()
        self._generation_done.set()

        self._msg_ch = utils.aio.Chan[dict[str, Any]]()
        self._input_resampler: rtc.AudioResampler | None = None
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    def _record_item(self, item: llm.ChatItem, *, remote: bool = False) -> None:
        """Track a completed conversation item so it can be replayed on reconnect.

        The realtime session's history lives on the provider and is lost when the
        socket is replaced, which happens on every proactive recycle
        (max_session_duration), not just on error. Keeping a local copy is also
        what makes ``chat_ctx`` return something to callers that read history off
        the session (``generate_reply(user_input=...)`` builds on it).

        ``remote`` marks an item the provider already holds because it produced
        it — its own reply, a tool call it asked for, a transcript from its own
        ASR. Those must not be sent back to it, and everything else must be.
        """
        if remote and item.id:
            self._remote_item_ids.add(item.id)
        if item.id and self._chat_ctx.get_by_id(item.id) is not None:
            return
        self._chat_ctx.items.append(item)

    @property
    def tools(self) -> llm.ToolContext:
        # The tools currently in force, which is what the pipeline saves and
        # restores around a per-turn override. Returning a stale set made that
        # save-and-restore clear the agent's tools for the rest of the call.
        return llm.ToolContext(self._current_tools)

    async def _connect(self) -> None:
        if self._started:
            return

        self._started = True
        try:
            await self._establish_ws()
        except Exception:
            # allow a later call to retry the initial connect
            self._started = False
            raise

        self._connected = True
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._send_task = asyncio.create_task(self._send_loop())

    async def _establish_ws(self) -> None:
        """Open the websocket and complete the session.create/session.created
        handshake.

        Shared by the initial connect and every reconnect attempt, so it only
        touches ``self._ws`` — task startup and ``_connected`` are owned by the
        caller (_connect for the first connect, _recv_loop's supervisor for
        reconnects).
        """
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()

        base_url = self._opts.base_url
        if base_url.startswith(("http://", "https://")):
            base_url = base_url.replace("http", "ws", 1)

        token = create_access_token(self._opts.api_key, self._opts.api_secret)
        headers = {
            **get_inference_headers(),
            "Authorization": f"Bearer {token}",
        }

        try:
            # asyncio.wait_for rather than ws_connect's own timeout=, matching
            # the STT and TTS siblings: aiohttp's kwarg takes a ClientWSTimeout,
            # where a bare number silently means something narrower than "give
            # up on the connect after this long".
            self._ws = await asyncio.wait_for(
                self._http_session.ws_connect(
                    f"{base_url}/sts?model={self._opts.model}",
                    headers=headers,
                ),
                _CONNECT_TIMEOUT,
            )
        except aiohttp.ClientResponseError as e:
            raise APIConnectionError(f"STS connection failed: {e.message}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("STS connection timed out") from e

        session_create: dict[str, Any] = {
            "type": "session.create",
            "model": self._opts.model,
            "voice": self._opts.voice,
            "modalities": self._opts.modalities,
            "instructions": self._opts.instructions,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
        }
        # Always send turn_detection so an explicit disable (None -> JSON null)
        # is forwarded. Omitting it makes the gateway apply its server_vad
        # default, which would silently re-enable turn detection the caller
        # disabled. A null value tells the realtime API to use manual turns.
        session_create["turn_detection"] = self._opts.turn_detection
        if self._opts.input_audio_transcription is not None:
            session_create["input_audio_transcription"] = self._opts.input_audio_transcription
        if self._opts.noise_reduction is not None:
            session_create["noise_reduction"] = self._opts.noise_reduction

        await self._ws.send_str(json.dumps(session_create))

        msg = await self._ws.receive(timeout=10)
        if msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.ERROR,
        ):
            raise APIConnectionError(f"STS connection closed during session creation: {msg.type}")

        if msg.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(msg.data)
            msg_type = data.get("type", "")
            if msg_type == "error":
                err_msg, err_code = _decode_error(data)
                raise APIError(f"STS session creation failed: {err_msg} (code={err_code})")
            if msg_type != "session.created":
                logger.warning("STS: expected session.created, got %s", msg_type)
        else:
            logger.warning("STS: unexpected message type during session creation: %s", msg.type)

    async def _recv_loop(self) -> None:
        # Supervises the read side across reconnects: read the current socket
        # until it drops, then try to re-establish the session unless we are
        # intentionally closing. Only after reconnect attempts are exhausted do we
        # surface a fatal error, so a transient network blip no longer kills the
        # session.
        while not self._closing:
            recycle_task: asyncio.Task | None = None
            if self._opts.max_session_duration is not None:
                recycle_task = asyncio.create_task(self._session_recycle_timer())

            try:
                await self._read_ws()
            finally:
                if recycle_task is not None:
                    recycle_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await recycle_task

            if self._closing:
                break

            if not await self._reconnect():
                self.emit(
                    "error",
                    llm.RealtimeModelError(
                        timestamp=time.time(),
                        label="sts",
                        error=APIConnectionError("STS connection closed unexpectedly"),
                        recoverable=False,
                    ),
                )
                break

        self._close_current_generation()

    async def _session_recycle_timer(self) -> None:
        # Proactively recycle before the provider's hard session cap. Wait for the
        # duration, then for any in-flight generation to finish, then close the
        # socket: that unblocks _read_ws, and the supervisor reconnects (this is a
        # reconnect, not a teardown, since _closing stays False). Cancelled by the
        # supervisor if the socket drops on its own first.
        assert self._opts.max_session_duration is not None
        await asyncio.sleep(self._opts.max_session_duration)
        await self._generation_done.wait()
        if self._ws is not None and not self._closing:
            logger.debug("STS: recycling session at max_session_duration")
            with contextlib.suppress(Exception):
                await self._ws.close()

    async def _reconnect(self) -> bool:
        """Re-establish a dropped session with bounded backoff.

        Returns True once a fresh socket is up and session state has been
        replayed, False if every attempt failed (or we started closing).
        """
        self._connected = False
        for attempt in range(_RECONNECT_MAX_RETRIES):
            if self._closing:
                return False
            try:
                await self._establish_ws()
            except Exception as e:
                logger.warning(
                    "STS: reconnect attempt %d/%d failed: %s",
                    attempt + 1,
                    _RECONNECT_MAX_RETRIES,
                    e,
                )
                if attempt < _RECONNECT_MAX_RETRIES - 1:
                    await asyncio.sleep(_RECONNECT_BASE_BACKOFF * (2**attempt))
                continue

            self._connected = True
            self._replay_session_state()
            logger.debug("STS: session reconnected")
            self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())
            return True

        return False

    def _replay_session_state(self) -> None:
        # The reconnected session starts fresh, so drop everything tied to the old
        # socket: fail in-flight response futures (their responses died with the
        # connection) and close the current generation so the pipeline isn't left
        # waiting on a dead turn.
        for fut in self._response_created_futures.values():
            if not fut.done():
                fut.set_exception(
                    llm.RealtimeError("pending response discarded due to session reconnection")
                )
        self._response_created_futures.clear()
        # A turn that was mid-flight is cut off here. Closing its channels quietly
        # would present the half-spoken reply to the pipeline as a finished one,
        # leaving the caller with a sentence that stops and then silence, with
        # nothing anywhere saying why. There is no re-speak hook for realtime
        # turns, so it is reported instead: an error the app can hear is the most
        # the SDK can honestly offer.
        if self._current_generation is not None:
            self.emit(
                "error",
                llm.RealtimeModelError(
                    timestamp=time.time(),
                    label="sts",
                    error=APIError("response interrupted by session reconnection"),
                    recoverable=True,
                ),
            )
        self._close_current_generation()
        self._input_transcripts.clear()
        self._output_transcripts.clear()
        self._truncated_transcripts.clear()
        self._discarded_event_ids.clear()
        # The new provider holds nothing until the replay below puts it back.
        self._remote_item_ids.clear()

        # session.create (in _establish_ws) already re-applied instructions, voice,
        # modalities and turn detection from _opts. Tools and tool_choice ride on
        # separate session.update events, so replay them onto the new session.
        if self._current_tools:
            self._queue_event(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "tools": _build_tool_defs(self._current_tools),
                    },
                }
            )
        if self._tool_choice != "auto":
            self._queue_event(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "tool_choice": self._tool_choice},
                }
            )

        # The provider's copy of the conversation died with the old socket, so
        # rebuild it from the locally tracked context. Without this the agent
        # forgets everything on every reconnect, including the proactive recycle
        # at max_session_duration, and may re-greet a caller mid-conversation.
        for item in self._chat_ctx.items:
            event = _chat_item_to_realtime_item(item)
            if event is None:
                continue
            self._queue_event({"type": "conversation.item.create", "item": event})
            self._remote_item_ids.add(item.id)

        # Only now: whatever the client asked for during the gap belongs after the
        # conversation it refers to.
        self._flush_deferred_events()

    async def _read_ws(self) -> None:
        if not self._ws:
            return

        while True:
            try:
                msg = await self._ws.receive()
            except Exception:
                break

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            ):
                break

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue

            self._handle_event(data)

    def _handle_event(self, data: dict[str, Any]) -> None:
        event_type = data.get("type", "")

        if event_type == "input_audio_buffer.speech_started":
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())

        elif event_type == "input_audio_buffer.speech_stopped":
            self.emit(
                "input_speech_stopped",
                llm.InputSpeechStoppedEvent(
                    user_transcription_enabled=self._opts.input_audio_transcription is not None
                ),
            )

        elif event_type == "conversation.item.input_audio_transcription.delta":
            self._handle_input_audio_transcription_delta(data)

        elif event_type == "conversation.item.input_audio_transcription.completed":
            self._handle_input_audio_transcription_completed(data)

        elif event_type == "conversation.item.input_audio_transcription.failed":
            self._handle_input_audio_transcription_failed(data)

        elif event_type == "response.created":
            self._handle_response_created(data)

        elif event_type == "response.output_item.added":
            self._handle_response_output_item_added(data)

        elif event_type == "response.content_part.added":
            self._handle_response_content_part_added(data)

        elif event_type == "response.output_audio.delta":
            self._handle_response_audio_delta(data)

        elif event_type == "response.output_audio_transcript.delta":
            self._handle_response_text_delta(data)

        elif event_type == "response.output_text.delta":
            self._handle_response_text_delta(data)

        elif event_type == "response.output_item.done":
            self._handle_response_output_item_done(data)

        elif event_type == "response.done":
            self._handle_response_done(data)

        elif event_type == "session.failover":
            self._handle_session_failover(data)

        elif event_type == "error":
            err_msg, err_code = _decode_error(data)
            # "Cancellation failed: no active response" is a benign race: an
            # interrupt() sent response.cancel just as the response ended, so
            # there is nothing to cancel and nothing to recover. OpenAI reports
            # it as an error; drop it instead of surfacing a scary log line
            # (mirrors the openai realtime plugin's _handle_error).
            if err_msg.startswith("Cancellation failed"):
                return
            logger.warning("STS error: %s (code=%s)", err_msg, err_code)
            self.emit(
                "error",
                llm.RealtimeModelError(
                    timestamp=time.time(),
                    label="sts_error",
                    error=APIError(err_msg),
                    # A quota, auth or billing failure will fail identically
                    # on every subsequent turn, so it is reported as fatal and
                    # the caller is spared a session that can only keep
                    # failing. Everything else is assumed transient.
                    recoverable=err_code not in _FATAL_ERROR_CODES,
                ),
            )

    def _handle_session_failover(self, data: dict[str, Any]) -> None:
        # The gateway moved the call to another deployment because the one
        # serving it died. Our socket is untouched, so this notice is the only
        # way to find out — and without acting on it the agent goes on talking
        # to a model that has no memory of the conversation.
        #
        # The gateway replays the session config it was given; everything
        # tracked on this side (tools, tool_choice, chat history) has to be
        # re-sent, which is the same work a dropped socket needs, so it shares
        # the reconnect path.
        if not data.get("context_lost", True):
            return

        logger.warning(
            "STS: provider failed over, replaying conversation onto the new session "
            "(model=%s, reason=%s)",
            data.get("model", ""),
            data.get("reason", ""),
        )
        self._replay_session_state()
        self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

    def _handle_input_audio_transcription_delta(self, data: dict[str, Any]) -> None:
        # OpenAI streams the user transcript incrementally as .delta events before
        # the final .completed. Accumulate per item so interim events carry the full
        # transcript so far (matching how the pipeline expects growing partials),
        # then emit a non-final transcription event for live captioning.
        item_id = data.get("item_id", "")
        delta = data.get("delta", "")
        if not item_id or not delta:
            return
        transcript = self._input_transcripts.get(item_id, "") + delta
        self._input_transcripts[item_id] = transcript
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=item_id,
                transcript=transcript,
                is_final=False,
            ),
        )

    def _handle_input_audio_transcription_completed(self, data: dict[str, Any]) -> None:
        # OpenAI transcribes the user's input audio when input_audio_transcription
        # is configured. The pipeline skips its own STT while user_transcription is
        # advertised, so this event is the only source of the user transcript; if
        # it isn't emitted the user's turn never lands in the transcript/history.
        item_id = data.get("item_id", "")
        self._input_transcripts.pop(item_id, None)
        transcript = data.get("transcript", "")
        if transcript:
            # The provider transcribed its own input audio, so the item is
            # already in its conversation.
            self._record_item(
                llm.ChatMessage(id=item_id, role="user", content=[transcript]), remote=True
            )
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=item_id,
                transcript=transcript,
                is_final=True,
            ),
        )

    def _handle_input_audio_transcription_failed(self, data: dict[str, Any]) -> None:
        # Transcription is best-effort: a failure means the user's turn has no
        # final transcript, but the audio turn itself is unaffected, so the
        # session carries on.
        #
        # Whatever partial arrived is closed out as final rather than dropped.
        # Captioning and history consumers have already been handed interim text
        # for this item and have no other way to learn it is finished; dropping it
        # leaves the last thing the user said displayed as a partial forever.
        # Matches the openai realtime plugin.
        item_id = data.get("item_id", "")
        partial = self._input_transcripts.pop(item_id, "")
        err = data.get("error", {})
        err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        logger.warning("STS: input audio transcription failed: %s", err_msg)

        if not partial:
            return
        self._record_item(llm.ChatMessage(id=item_id, role="user", content=[partial]), remote=True)
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id=item_id, transcript=partial, is_final=True),
        )

    def _handle_response_created(self, data: dict[str, Any]) -> None:
        response = data.get("response", {})
        response_id = response.get("id", "")

        metadata = response.get("metadata", {})
        client_event_id = metadata.get("client_event_id", "") if isinstance(metadata, dict) else ""
        if client_event_id and client_event_id in self._discarded_event_ids:
            # The caller interrupted before this reply existed. Cancel it again —
            # the first cancel raced the creation — and don't announce it, so the
            # pipeline never gets a turn it already abandoned.
            self._discarded_event_ids.discard(client_event_id)
            self._queue_event({"type": "response.cancel"})
            return

        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            messages={},
            response_id=response_id,
            created_timestamp=time.time(),
        )
        # a turn is now in flight; hold off any proactive session recycle
        self._generation_done.clear()

        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
            response_id=response_id,
        )

        if client_event_id and client_event_id in self._response_created_futures:
            fut = self._response_created_futures.pop(client_event_id)
            if not fut.done():
                generation_ev.user_initiated = True
                fut.set_result(generation_ev)

        self.emit("generation_created", generation_ev)

    def _handle_response_output_item_added(self, data: dict[str, Any]) -> None:
        if self._current_generation is None:
            return

        item = data.get("item", {})
        item_id = item.get("id", "")
        item_type = item.get("type", "")

        if item_type == "message":
            item_gen = _MessageGeneration(
                message_id=item_id,
                text_ch=utils.aio.Chan(),
                audio_ch=utils.aio.Chan(),
                modalities=asyncio.get_running_loop().create_future(),
            )
            self._current_generation.messages[item_id] = item_gen

            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item_id,
                    text_stream=item_gen.text_ch,
                    audio_stream=item_gen.audio_ch,
                    modalities=item_gen.modalities,
                )
            )

    def _handle_response_content_part_added(self, data: dict[str, Any]) -> None:
        if self._current_generation is None:
            return

        item_id = data.get("item_id", "")
        part = data.get("part", {})
        part_type = part.get("type", "")

        item_gen = self._current_generation.messages.get(item_id)
        if item_gen and not item_gen.modalities.done():
            if part_type == "audio":
                item_gen.modalities.set_result(["audio", "text"])
            elif part_type == "text":
                item_gen.modalities.set_result(["text"])

    def _handle_response_audio_delta(self, data: dict[str, Any]) -> None:
        if self._current_generation is None:
            return

        item_id = data.get("item_id", "")
        item_gen = self._current_generation.messages.get(item_id)
        if item_gen is None:
            return

        if self._current_generation.first_token_timestamp is None:
            self._current_generation.first_token_timestamp = time.time()

        if not item_gen.modalities.done():
            item_gen.modalities.set_result(["audio", "text"])

        delta = data.get("delta", "")
        if delta:
            audio_data = base64.b64decode(delta)
            item_gen.audio_ch.send_nowait(
                rtc.AudioFrame(
                    data=audio_data,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=len(audio_data) // 2,
                )
            )

    def _handle_response_text_delta(self, data: dict[str, Any]) -> None:
        if self._current_generation is None:
            return

        if self._current_generation.first_token_timestamp is None:
            self._current_generation.first_token_timestamp = time.time()

        item_id = data.get("item_id", "")
        item_gen = self._current_generation.messages.get(item_id)
        if item_gen is None:
            return

        delta = data.get("delta", "")
        if delta:
            item_gen.text_ch.send_nowait(delta)
            self._output_transcripts[item_id] = self._output_transcripts.get(item_id, "") + delta

    def _handle_response_output_item_done(self, data: dict[str, Any]) -> None:
        if self._current_generation is None:
            return

        item = data.get("item", {})
        item_id = item.get("id", "")
        item_type = item.get("type", "")

        if item_type == "function_call":
            # Arguments stream in after output_item.added and are only complete here.
            call_id = item.get("call_id", "")
            name = item.get("name", "")
            arguments = item.get("arguments", "")
            if call_id and name:
                fnc = llm.FunctionCall(
                    id=item_id,
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                )
                self._record_item(fnc, remote=True)
                self._current_generation.function_ch.send_nowait(fnc)
            return

        # An interrupted turn was already trimmed to what the caller heard; the
        # accumulated transcript includes words that were generated but never
        # played.
        transcript = self._output_transcripts.pop(item_id, "")
        if item_id in self._truncated_transcripts:
            transcript = self._truncated_transcripts.pop(item_id)
        if transcript:
            self._record_item(
                llm.ChatMessage(id=item_id, role="assistant", content=[transcript]), remote=True
            )

        item_gen = self._current_generation.messages.get(item_id)
        if item_gen:
            if not item_gen.text_ch.closed:
                item_gen.text_ch.close()
            if not item_gen.audio_ch.closed:
                item_gen.audio_ch.close()
            if not item_gen.modalities.done():
                item_gen.modalities.set_result(self._model._opts.modalities)

    def _handle_response_done(self, data: dict[str, Any]) -> None:
        self._emit_usage_metrics(data)
        # The channels close either way — leaving them open would hang the
        # pipeline on a turn that is over — but a failed or truncated response is
        # reported, because closing quietly makes a rate-limited turn
        # indistinguishable from one where the model chose to say nothing.
        # Matches the openai realtime plugin.
        self._report_incomplete_response(data.get("response", {}))
        self._close_current_generation()

    def _report_incomplete_response(self, response: dict[str, Any]) -> None:
        status = response.get("status", "")
        if status in ("", "completed", "cancelled"):
            # A cancellation is an interruption the pipeline asked for.
            return

        details = response.get("status_details") or {}
        message = f"STS response {status}"
        if isinstance(details, dict):
            error = details.get("error") or {}
            if isinstance(error, dict) and (error.get("code") or error.get("type")):
                message = f"{message}: [{error.get('type', '')}] {error.get('code', '')}".strip()
            elif details.get("reason"):
                message = f"{message}: {details['reason']}"

        logger.warning("STS: %s", message)
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label="sts",
                error=APIError(message),
                recoverable=True,
            ),
        )

    def _emit_usage_metrics(self, data: dict[str, Any]) -> None:
        response = data.get("response", {})
        usage = response.get("usage", {})
        if not usage:
            return

        gen = self._current_generation
        created_timestamp = gen.created_timestamp if gen else time.time()
        first_token_timestamp = gen.first_token_timestamp if gen else None
        response_id = response.get("id", gen.response_id if gen else "")
        status = response.get("status", "")

        ttft = first_token_timestamp - created_timestamp if first_token_timestamp else -1
        duration = time.time() - created_timestamp

        input_details = usage.get("input_token_details", {})
        output_details = usage.get("output_token_details", {})
        cached_details = input_details.get("cached_tokens_details", {})

        metrics = RealtimeModelMetrics(
            timestamp=created_timestamp,
            request_id=response_id,
            ttft=ttft,
            duration=duration,
            cancelled=status == "cancelled",
            label="sts",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            tokens_per_second=(usage.get("output_tokens", 0) / duration if duration > 0 else 0),
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                audio_tokens=input_details.get("audio_tokens", 0),
                cached_tokens=input_details.get("cached_tokens", 0),
                text_tokens=input_details.get("text_tokens", 0),
                cached_tokens_details=RealtimeModelMetrics.CachedTokenDetails(
                    text_tokens=cached_details.get("text_tokens", 0),
                    audio_tokens=cached_details.get("audio_tokens", 0),
                    image_tokens=cached_details.get("image_tokens", 0),
                ),
                image_tokens=input_details.get("image_tokens", 0),
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=output_details.get("text_tokens", 0),
                audio_tokens=output_details.get("audio_tokens", 0),
                image_tokens=output_details.get("image_tokens", 0),
            ),
            metadata=Metadata(
                model_name=self._model._opts.model,
                model_provider="livekit",
            ),
        )
        self.emit("metrics_collected", metrics)

    def _close_current_generation(self) -> None:
        if self._current_generation is None:
            # no active turn; make sure a recycle waiter isn't left blocked
            self._generation_done.set()
            return

        for item_gen in self._current_generation.messages.values():
            if not item_gen.text_ch.closed:
                item_gen.text_ch.close()
            if not item_gen.audio_ch.closed:
                item_gen.audio_ch.close()
            if not item_gen.modalities.done():
                item_gen.modalities.set_result(self._model._opts.modalities)

        if not self._current_generation.function_ch.closed:
            self._current_generation.function_ch.close()
        if not self._current_generation.message_ch.closed:
            self._current_generation.message_ch.close()

        self._current_generation = None
        # turn finished; a pending session recycle may now proceed
        self._generation_done.set()

    async def _send_loop(self) -> None:
        # Runs for the lifetime of the session across reconnects.
        #
        # While the socket is down, events are set aside instead of discarded: a
        # reply the pipeline started during the reconnect window would otherwise
        # be dropped and then failed, which reads to the caller as the agent going
        # silent for a turn. They are re-queued *after* the replay
        # (_replay_session_state) rather than held in place, so the replacement
        # sees the conversation before it is asked to answer.
        #
        # Audio is the exception. It is real-time by nature: a second of speech
        # delivered after the gap is worse than no speech, and it would arrive
        # ahead of the history for the same reason.
        async for msg in self._msg_ch:
            if not self._connected or not self._ws:
                self._defer_event(msg)
                continue
            try:
                await self._ws.send_str(json.dumps(msg))
            except Exception:
                if not self._closing:
                    logger.warning("STS: failed to send event, connection closed")

    def _defer_event(self, event: dict[str, Any]) -> None:
        """Hold an event the current socket can no longer carry."""
        if self._closing:
            return
        event_type = event.get("type", "")
        if event_type.startswith("input_audio_buffer."):
            return
        if len(self._deferred_events) >= _MAX_DEFERRED_EVENTS:
            # The reconnect budget is a few seconds; anything longer than this
            # backlog means the session is not coming back, and replaying a
            # minute of stale requests onto it would be worse than losing them.
            logger.warning("STS: dropping event queued during reconnect, backlog full")
            return
        self._deferred_events.append(event)

    def _flush_deferred_events(self) -> None:
        """Re-queue what the reconnect window held, behind the replayed state."""
        deferred, self._deferred_events = self._deferred_events, []
        for event in deferred:
            self._queue_event(event)

    def _queue_event(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    async def _send(self, event: dict[str, Any]) -> None:
        # Only start the lifecycle once. During a reconnect (_started True but
        # _connected briefly False) just queue; the send pump flushes once the
        # socket is back, and dropped mid-gap events are re-applied by
        # _replay_session_state where it matters (instructions/tools/tool_choice).
        if not self._started:
            await self._connect()
        self._queue_event(event)

    async def update_instructions(self, instructions: str) -> None:
        self._opts.instructions = instructions
        await self._send(
            {
                "type": "session.update",
                "session": {"type": "realtime", "instructions": instructions},
            }
        )

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        # Everything the provider has not been told about goes over: the history
        # a caller preloaded into AgentSession, the text of
        # generate_reply(user_input=...), and the tool outputs a turn is waiting
        # on. What the model produced itself is skipped — it already has those,
        # and re-sending them would duplicate the turn it just took.
        #
        # This is deliberately additive: the provider owns the live conversation,
        # so removals and edits are not replayed the way the OpenAI plugin's
        # diff does. The one edit that matters (an interrupted reply trimmed to
        # what was actually played) is handled in truncate.
        for item in chat_ctx.items:
            self._record_item(item)

            if item.id in self._remote_item_ids:
                continue
            if isinstance(item, llm.FunctionCallOutput) and item.call_id in self._sent_fnc_outputs:
                self._remote_item_ids.add(item.id)
                continue

            event = _chat_item_to_realtime_item(item)
            if event is None:
                continue
            if isinstance(item, llm.FunctionCallOutput):
                self._sent_fnc_outputs.add(item.call_id)
            self._remote_item_ids.add(item.id)
            await self._send({"type": "conversation.item.create", "item": event})

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        # Retain the tools so _replay_session_state can re-apply them after a
        # reconnect (session.create does not carry tools).
        self._current_tools = list(tools)
        await self._send(
            {
                "type": "session.update",
                "session": {"type": "realtime", "tools": _build_tool_defs(tools)},
            }
        )

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if not is_given(tool_choice):
            return
        # Session-level tool selection: forward as a session.update so it applies
        # to subsequent turns. The voice pipeline uses this path when it is not
        # overriding tool_choice per response (see agent_activity). Only emit on a
        # real change to avoid a redundant session.update every turn. Mirrors the
        # openai realtime plugin's update_options.
        new_choice = _to_realtime_tool_choice(tool_choice)
        if new_choice == self._tool_choice:
            return
        self._tool_choice = new_choice
        self._queue_event(
            {
                "type": "session.update",
                "session": {"type": "realtime", "tool_choice": new_choice},
            }
        )

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._connected or not self._ws:
            logger.warning("STS push_audio called before session is connected, dropping frame")
            return
        for f in self._resample_audio(frame):
            data = f.data.tobytes()
            for nf in self._bstream.write(data):
                self._queue_event(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(nf.data).decode("utf-8"),
                    }
                )

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        event_id = utils.shortuuid("response_create_")
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.get_running_loop().create_future()
        self._response_created_futures[event_id] = fut

        response_params: dict[str, Any] = {
            "metadata": {"client_event_id": event_id},
        }
        if is_given(instructions):
            response_params["instructions"] = instructions
        # tool_choice/tools are part of the RealtimeSession.generate_reply contract
        # (llm/realtime.py) and the voice pipeline always passes them
        # (agent_activity.py). Forward them into response.create, mirroring the
        # openai realtime plugin's generate_reply, so per-response tool selection
        # works instead of raising on the unexpected kwargs.
        if is_given(tool_choice):
            response_params["tool_choice"] = _to_realtime_tool_choice(tool_choice)
        if is_given(tools):
            response_params["tools"] = _build_tool_defs(tools)

        self._queue_event(
            {
                "type": "response.create",
                "event_id": event_id,
                "response": response_params,
            }
        )

        def _on_timeout() -> None:
            self._response_created_futures.pop(event_id, None)
            if not fut.done():
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))

        handle = asyncio.get_running_loop().call_later(10.0, _on_timeout)

        def _on_done(f: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            handle.cancel()
            self._response_created_futures.pop(event_id, None)
            if not f.cancelled():
                return
            # The pipeline cancels this when the caller interrupts before the
            # reply started. The provider hasn't heard about that, so without
            # cancelling upstream it goes on to produce the reply and speak it
            # over whatever the caller said instead. The id is remembered because
            # response.created can already be in flight: when it lands it is
            # discarded rather than emitted as a fresh unprompted turn.
            self._discarded_event_ids.add(event_id)
            self._queue_event({"type": "response.cancel"})

        fut.add_done_callback(_on_done)

        return fut

    def commit_audio(self) -> None:
        if self._ws:
            for nf in self._bstream.flush():
                self._queue_event(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(nf.data).decode("utf-8"),
                    }
                )
            self._queue_event({"type": "input_audio_buffer.commit"})

    def clear_audio(self) -> None:
        if self._ws:
            self._queue_event({"type": "input_audio_buffer.clear"})

    @property
    def has_active_generation(self) -> bool:
        return self._current_generation is not None or len(self._response_created_futures) > 0

    def interrupt(self) -> None:
        if not self._ws or not self.has_active_generation:
            return
        self._queue_event({"type": "response.cancel"})

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if self._ws:
            self._queue_event(
                {
                    "type": "conversation.item.truncate",
                    "item_id": message_id,
                    "content_index": 0,
                    "audio_end_ms": audio_end_ms,
                }
            )

        # The caller heard audio_transcript, not the whole reply the model
        # generated. Trim the local record to what was actually played, because
        # that record is what gets replayed onto a replacement provider: left
        # alone, a recycle or failover would restore words the caller interrupted
        # and never heard, and the model would answer as if it had said them.
        #
        # Remembered as well as applied, because the interrupted item may not be
        # recorded yet: the pipeline truncates when playback stops, which can
        # precede the response.output_item.done that records the turn.
        if not is_given(audio_transcript):
            return
        self._truncated_transcripts[message_id] = audio_transcript
        idx = self._chat_ctx.index_by_id(message_id)
        if idx is None:
            return
        item = self._chat_ctx.items[idx]
        if isinstance(item, llm.ChatMessage):
            item.content = [audio_transcript]

    async def aclose(self) -> None:
        self._closing = True
        # Fail anything still waiting on a reply. Left alone these sit until their
        # ten-second timeout, holding the caller that awaited generate_reply well
        # past the point the session stopped existing.
        for fut in self._response_created_futures.values():
            if not fut.done():
                fut.set_exception(llm.RealtimeError("session closed"))
        self._response_created_futures.clear()
        self._close_current_generation()
        self._msg_ch.close()
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
        if self._http_session:
            await self._http_session.close()
        self._connected = False
