import asyncio
import base64
import copy
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


SRC = Path(__file__).resolve().parents[1] / "src" / "resilient-copilot"
sys.path.insert(0, str(SRC))

import agent
from azure.ai.agentserver.core.storage import FoundryStateStore
from azure.ai.agentserver.core.tasks import TaskContext
from copilot._jsonrpc import JsonRpcClient
from copilot.session import CopilotSession
from copilot.session_events import SessionEventType
from copilot.session_events import SessionEvent
from copilot_session import (
    CopilotHarness,
    CopilotSessionError,
    ElicitationDecisionRequest,
    ElicitationRequest,
    HarnessClosingError,
    IdleBoundary,
    NativeInputRequest,
    TodoCheckpointRequest,
)
import store as snapshot_module
from store import (
    SessionSnapshotStore,
    SnapshotTooLargeError,
    _pack_dir,
    _unpack_dir,
    session_storage_key,
    validate_session_id,
)
import main


class SnapshotSafetyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_session_ids_are_validated_and_filesystem_keys_are_hashed(self):
        valid = "A_b-9" * 20
        self.assertEqual(validate_session_id(valid), valid)
        key = session_storage_key(valid)
        self.assertEqual(len(key), 64)
        self.assertNotIn(valid, key)
        for invalid in (
            "",
            "../escape",
            r"..\escape",
            "/absolute",
            r"C:\absolute",
            "has/slash",
            "has\\slash",
            "x" * 121,
            "é",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_session_id(invalid)

    def test_archive_rejects_traversal_before_replacing_destination(self):
        destination = self.root / "destination"
        destination.mkdir()
        (destination / "keep.txt").write_text("keep", encoding="utf-8")
        blob = io.BytesIO()
        with tarfile.open(fileobj=blob, mode="w:gz") as archive:
            info = tarfile.TarInfo("../escape.txt")
            info.size = 3
            archive.addfile(info, io.BytesIO(b"bad"))
        with self.assertRaisesRegex(ValueError, "Unsafe snapshot"):
            _unpack_dir(blob.getvalue(), destination)
        self.assertEqual((destination / "keep.txt").read_text(), "keep")
        self.assertFalse((self.root / "escape.txt").exists())

    def test_archive_restores_legacy_windows_member_names_safely(self):
        blob = io.BytesIO()
        with tarfile.open(fileobj=blob, mode="w:gz") as archive:
            info = tarfile.TarInfo(r"session-state\test\value.txt")
            info.size = 2
            archive.addfile(info, io.BytesIO(b"ok"))
        destination = self.root / "destination"
        _unpack_dir(blob.getvalue(), destination)
        self.assertEqual(
            (destination / "session-state" / "test" / "value.txt").read_text(),
            "ok",
        )

    async def test_local_restore_reads_legacy_filename_into_safe_directory(self):
        source = self.root / "source"
        source.mkdir()
        (source / "session-store.db").write_bytes(b"db")
        snapshots = SessionSnapshotStore(self.root / "snapshots")
        snapshots._use_state_store = False
        snapshots._legacy_path("legacy").write_bytes(_pack_dir(source))
        destination = self.root / session_storage_key("legacy")
        self.assertTrue(await snapshots.restore("legacy", destination))
        self.assertEqual((destination / "session-store.db").read_bytes(), b"db")

    async def test_state_store_allows_two_mib_and_rejects_over_five_mib(self):
        config = self.root / "config"
        config.mkdir()
        snapshots = SessionSnapshotStore(self.root / "snapshots")
        snapshots._use_state_store = True
        cloud = SimpleNamespace(set_item=AsyncMock())
        snapshots._store = cloud
        with patch.object(
            snapshot_module, "_pack_dir", return_value=b"x" * (2 * 1024 * 1024)
        ):
            await snapshots.save("test", config)
        cloud.set_item.assert_awaited_once()

        cloud.set_item.reset_mock()
        with patch.object(
            snapshot_module, "_pack_dir", return_value=b"x" * (4 * 1024 * 1024)
        ), self.assertRaisesRegex(SnapshotTooLargeError, "5 MiB serialized JSON"):
            await snapshots.save("test", config)
        cloud.set_item.assert_not_awaited()

    def test_live_jsonl_prefix_is_streamed_without_read_bytes(self):
        config = self.root / "config"
        config.mkdir()
        complete = b'{"type":"saved"}\n' * 10_000
        (config / "events.jsonl").write_bytes(complete + b'{"partial":')

        with patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("live JSONL must be streamed"),
        ):
            blob = _pack_dir(config, live=True)

        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
            restored = archive.extractfile("events.jsonl")
            self.assertIsNotNone(restored)
            self.assertEqual(restored.read(), complete)

    async def test_oversized_pack_preserves_previous_cloud_snapshot(self):
        old_config = self.root / "old-config"
        old_config.mkdir()
        (old_config / "session-store.db").write_bytes(b"old snapshot")
        old_value = {
            "tar_gz_b64": base64.b64encode(_pack_dir(old_config)).decode("ascii")
        }

        config = self.root / "config"
        config.mkdir()
        with (config / "large.bin").open("wb") as large:
            for _ in range(80):
                large.write(os.urandom(64 * 1024))

        cloud = SimpleNamespace(
            set_item=AsyncMock(),
            get_item=AsyncMock(return_value=SimpleNamespace(value=old_value)),
        )
        snapshots = SessionSnapshotStore(self.root / "snapshots")
        snapshots._use_state_store = True
        snapshots._store = cloud

        with self.assertRaisesRegex(SnapshotTooLargeError, "5 MiB serialized JSON"):
            await snapshots.save("test", config)
        cloud.set_item.assert_not_awaited()

        restored = self.root / "restored"
        self.assertTrue(await snapshots.restore("test", restored))
        self.assertEqual((restored / "session-store.db").read_bytes(), b"old snapshot")

    async def test_versioned_store_uses_thirty_days_and_reads_legacy_cloud_item(self):
        snapshots = SessionSnapshotStore(self.root / "snapshots")
        snapshots._use_state_store = True
        created = SimpleNamespace(get_item=AsyncMock(return_value=None))
        legacy_blob = _pack_dir(self.root)
        legacy = SimpleNamespace(
            get_item=AsyncMock(
                return_value=SimpleNamespace(
                    value={"tar_gz_b64": base64.b64encode(legacy_blob).decode("ascii")}
                )
            )
        )
        snapshots._store = created
        snapshots._legacy_store = legacy
        destination = self.root / "restored"
        self.assertTrue(await snapshots.restore("test", destination))
        legacy.get_item.assert_awaited_once_with("test")

        snapshots = SessionSnapshotStore(self.root / "snapshots-2")
        snapshots._use_state_store = True
        resolved = SimpleNamespace()
        with patch.object(
            FoundryStateStore,
            "get_or_create",
            AsyncMock(return_value=resolved),
        ) as get_or_create:
            self.assertIs(await snapshots._state_store(), resolved)
        self.assertEqual(
            get_or_create.call_args.kwargs["item_ttl_seconds"],
            30 * 24 * 60 * 60,
        )
        self.assertEqual(
            get_or_create.call_args.args[0], "copilot-session-snapshots-v2"
        )

    async def test_missing_legacy_store_is_absent_with_real_local_backend(self):
        state_root = self.root / "real-core-state"
        destination = self.root / "fresh-config"
        with patch.dict(
            os.environ,
            {
                "AGENTSERVER_STATE_ROOT": str(state_root),
                "FOUNDRY_PROJECT_ENDPOINT": "https://example.test",
                "FOUNDRY_HOSTING_ENVIRONMENT": "",
            },
            clear=False,
        ):
            snapshots = SessionSnapshotStore(self.root / "snapshots-real")
            self.assertTrue(snapshots._use_state_store)
            self.assertFalse(await snapshots.restore("fresh", destination))
            await snapshots.aclose()

        self.assertFalse(destination.exists())
        documents = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in state_root.rglob("*.json")
        ]
        store_names = {document["store"]["name"] for document in documents}
        self.assertIn("copilot-session-snapshots-v2", store_names)
        self.assertNotIn("copilot-session-snapshots", store_names)


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_bearer_token_provider_refreshes_and_credential_closes(self):
        credential = SimpleNamespace(
            get_token=AsyncMock(
                side_effect=[
                    SimpleNamespace(token="first"),
                    SimpleNamespace(token="second"),
                ]
            ),
            close=AsyncMock(),
        )
        harness = CopilotHarness()
        with patch.dict(
            os.environ,
            {
                "FOUNDRY_PROJECT_ENDPOINT": "https://example.test",
                "AZURE_AI_MODEL_DEPLOYMENT_NAME": "model",
            },
            clear=True,
        ), patch("azure.identity.aio.DefaultAzureCredential", return_value=credential):
            provider, model = harness._byok_provider()
        self.assertEqual(model, "model")
        callback = provider["bearer_token_provider"]
        self.assertEqual(await callback({}), "first")
        self.assertEqual(await callback({}), "second")
        self.assertEqual(credential.get_token.await_count, 2)
        await harness.close()
        credential.close.assert_awaited_once()

    async def test_resume_failure_never_creates_fresh_session(self):
        harness = CopilotHarness()
        client = SimpleNamespace(
            start=AsyncMock(),
            resume_session=AsyncMock(side_effect=OSError("corrupt session")),
            create_session=AsyncMock(),
            stop=AsyncMock(),
        )
        with patch("copilot_session.CopilotClient", return_value=client), patch.dict(
            os.environ, {"GITHUB_TOKEN": "test"}, clear=True
        ), patch.object(harness, "_byok_provider", return_value=(None, None)):
            with self.assertRaisesRegex(OSError, "corrupt session"):
                await harness.start("test", recover=True)
        client.create_session.assert_not_awaited()
        client.stop.assert_awaited_once()

    async def test_session_error_is_forwarded_then_terminates_iterator(self):
        harness = CopilotHarness()
        event = SimpleNamespace(type=SessionEventType.SESSION_ERROR, data="failed")
        harness._on_event(event)
        events = harness.events()
        self.assertIs(await anext(events), event)
        with self.assertRaisesRegex(CopilotSessionError, "failed"):
            await anext(events)

    async def test_close_rejects_pending_control(self):
        harness = CopilotHarness()
        pending = asyncio.create_task(harness.queue_input("input", "prompt", "enqueue"))
        await asyncio.sleep(0)
        await harness.close()
        with self.assertRaises(HarnessClosingError):
            await pending

    async def test_pinned_dispatcher_orders_idle_around_send_response(self):
        async def exercise(order):
            harness = CopilotHarness()
            loop = asyncio.get_running_loop()
            rpc = JsonRpcClient(
                SimpleNamespace(
                    stdin=SimpleNamespace(write=lambda _data: None, flush=lambda: None)
                )
            )
            rpc._loop = loop
            idle = SimpleNamespace(type=SessionEventType.SESSION_IDLE)
            ordinary = SimpleNamespace(type=SessionEventType.ASSISTANT_INTENT)
            rpc.set_notification_handler(
                lambda _method, params: harness._on_event(
                    idle if params["kind"] == "idle" else ordinary
                )
            )
            harness.session = SimpleNamespace(
                _client=rpc,
                session_id="session",
                abort=AsyncMock(),
            )
            sending = asyncio.create_task(
                harness.send("follow up", mode="enqueue", causal_idle=True)
            )
            while not rpc.pending_requests:
                await asyncio.sleep(0)
            request_id = next(iter(rpc.pending_requests))

            messages = {
                "ordinary": {
                    "jsonrpc": "2.0",
                    "method": "session.event",
                    "params": {"kind": "ordinary"},
                },
                "idle": {
                    "jsonrpc": "2.0",
                    "method": "session.event",
                    "params": {"kind": "idle"},
                },
                "response": {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"messageId": "message-id"},
                },
            }
            thread = threading.Thread(
                target=lambda: [rpc._handle_message(messages[name]) for name in order]
            )
            thread.start()
            await asyncio.to_thread(thread.join)
            self.assertEqual(await asyncio.wait_for(sending, 2), "message-id")
            return harness, idle, ordinary

        stale_harness, stale_idle, ordinary = await exercise(
            ("ordinary", "idle", "response")
        )
        self.assertIs(
            await asyncio.wait_for(anext(stale_harness.events()), 2), ordinary
        )
        waiting = asyncio.create_task(anext(stale_harness.events()))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        stale_harness._on_event(stale_idle)
        self.assertIs(await asyncio.wait_for(waiting, 2), stale_idle)

        current_harness, current_idle, _ = await exercise(("response", "idle"))
        self.assertIs(
            await asyncio.wait_for(anext(current_harness.events()), 2),
            current_idle,
        )

    async def test_pinned_elicitation_dispatcher_waits_for_consumer_ack(self):
        response_started = asyncio.Event()
        response_release = asyncio.Event()

        class Client:
            async def request(self, method, params):
                response_started.set()
                await response_release.wait()
                return {"success": True}

        harness = CopilotHarness()
        sdk_session = CopilotSession("session", Client())
        sdk_session._elicitation_handler = harness._elicitation_handler
        event = SessionEvent.from_dict(
            {
                "type": "elicitation.requested",
                "id": "00000000-0000-0000-0000-000000000001",
                "timestamp": "2026-10-12T00:00:00Z",
                "data": {
                    "requestId": "request-1",
                    "message": "Continue?",
                },
            }
        )
        sdk_session._dispatch_event(event)

        request = await asyncio.wait_for(harness._queue.get(), 2)
        self.assertIsInstance(request, ElicitationRequest)
        self.assertFalse(response_started.is_set())
        request.acknowledged.set_result(None)
        self.assertTrue(
            harness.answer_elicitation(request.elicitation_id, "accept", {})
        )
        await asyncio.wait_for(response_started.wait(), 2)
        self.assertTrue(
            any(
                "_handle_elicitation_request" in repr(task.get_coro())
                for task in asyncio.all_tasks()
                if not task.done()
            )
        )
        response_release.set()
        for _ in range(20):
            await asyncio.sleep(0)
            if not any(
                "_handle_elicitation_request" in repr(task.get_coro())
                for task in asyncio.all_tasks()
                if not task.done()
            ):
                break


class FakeStream:
    def __init__(self):
        self.frames = []
        self.close = AsyncMock()

    async def last_cursor(self):
        return 0

    async def emit(self, frame):
        self.frames.append(frame)


class FakeStateStore:
    def __init__(self, value=None, log=None):
        self.value = copy.deepcopy(value)
        self.log = log
        self.aclose = AsyncMock()

    async def get_item(self, key):
        if self.value is None:
            return None
        return SimpleNamespace(value=copy.deepcopy(self.value))

    async def set_item(self, key, value):
        self.value = copy.deepcopy(value)
        if self.log is not None:
            statuses = [entry.get("status") for entry in value.get("input_journal", [])]
            self.log.append(f"state:{','.join(statuses)}")


class QueueHarness:
    instances = []
    start_gate = None
    log = None

    def __init__(self):
        self.queue = asyncio.Queue()
        self.pending = set()
        self.accepting = True
        self.sent = []
        self.start = AsyncMock(side_effect=self._start)
        self.abort = AsyncMock(side_effect=self._abort)
        self.close = AsyncMock(side_effect=self._close)
        self.idle_seen = asyncio.Event()
        self.live_elicitation = False
        type(self).instances.append(self)

    async def _start(self, *args, **kwargs):
        if type(self).start_gate is not None:
            await type(self).start_gate.wait()

    async def send(self, prompt, *, mode=None, causal_idle=False):
        self.sent.append((prompt, mode))
        if type(self).log is not None:
            type(self).log.append(f"send:{prompt}")
        return "message"

    async def events(self):
        while True:
            item = await self.queue.get()
            if item is None:
                self.idle_seen.set()
                return
            yield item

    async def queue_input(self, input_id, prompt, mode):
        if not self.accepting:
            raise HarnessClosingError()
        request = NativeInputRequest(
            input_id, prompt, mode, asyncio.get_running_loop().create_future()
        )
        self.pending.add(request.acknowledged)
        await self.queue.put(request)
        try:
            return await request.acknowledged
        finally:
            self.pending.discard(request.acknowledged)

    async def queue_elicitation_decision(self, elicitation_id, action, content):
        if not self.accepting:
            raise HarnessClosingError()
        request = ElicitationDecisionRequest(
            elicitation_id,
            action,
            content,
            asyncio.get_running_loop().create_future(),
        )
        self.pending.add(request.acknowledged)
        await self.queue.put(request)
        try:
            return await request.acknowledged
        finally:
            self.pending.discard(request.acknowledged)

    def answer_elicitation(self, elicitation_id, action, content):
        return self.live_elicitation

    def bind_recovered_elicitations(self, markers):
        self.bound_markers = markers

    def discard_recovered_elicitation(self, elicitation_id):
        return None

    def stop_accepting_controls(self):
        self.accepting = False
        for pending in self.pending:
            if not pending.done():
                pending.set_exception(HarnessClosingError())

    async def _abort(self):
        return None

    async def _close(self):
        self.stop_accepting_controls()


class ImmediateHarness(QueueHarness):
    async def events(self):
        if False:
            yield None


class AgentTaskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        agent._HARNESS_SLOTS.clear()
        agent._CLIENT_CANCEL_REQUESTED.clear()
        QueueHarness.instances.clear()
        QueueHarness.start_gate = None
        QueueHarness.log = None
        ImmediateHarness.log = None

    def context(
        self,
        *,
        entry_mode="fresh",
        cancel=False,
        shutdown=False,
        cancel_requested=False,
    ):
        cancel_event = asyncio.Event()
        shutdown_event = asyncio.Event()
        if cancel:
            cancel_event.set()
        if shutdown:
            shutdown_event.set()
        return SimpleNamespace(
            input={
                "invocation_id": "invocation",
                "prompt": "initial",
                "session_id": "session",
            },
            entry_mode=entry_mode,
            recovery_count=1 if entry_mode == "recovered" else 0,
            cancel=cancel_event,
            shutdown=shutdown_event,
            cancel_requested=cancel_requested,
            exit_for_recovery=AsyncMock(return_value="deferred"),
        )

    async def run_task(
        self,
        *,
        context=None,
        state=None,
        harness=ImmediateHarness,
        restore=False,
        restore_side_effect=None,
        snapshot_side_effect=None,
        log=None,
    ):
        context = context or self.context()
        stream = FakeStream()
        state_store = FakeStateStore(state, log)
        snapshot = SimpleNamespace(
            restore=AsyncMock(return_value=restore),
            save=AsyncMock(side_effect=snapshot_side_effect),
        )
        if restore_side_effect is not None:
            snapshot.restore.side_effect = restore_side_effect
        with patch.object(agent, "_CONFIG_ROOT", self.root / "config"), patch.object(
            agent, "_SNAPSHOT_STORE", snapshot
        ), patch.object(agent, "CopilotHarness", harness), patch.object(
            agent.streams, "get_or_create", AsyncMock(return_value=stream)
        ), patch.object(
            agent.FoundryStateStore,
            "get_or_create",
            AsyncMock(return_value=state_store),
        ), patch.object(
            agent, "_get_legacy_state_item", AsyncMock(return_value=None)
        ):
            result = await agent.copilot_agent._fn(context)
        return result, stream, state_store, snapshot

    async def test_missing_local_db_restores_on_resumed_entry(self):
        async def restore(_session_id, config_dir):
            config_dir.mkdir(parents=True)
            (config_dir / "session-store.db").write_bytes(b"db")
            return True

        await self.run_task(
            context=self.context(entry_mode="resumed"),
            state={"has_snapshot": True},
            restore_side_effect=restore,
        )
        harness = QueueHarness.instances[-1]
        self.assertTrue(harness.start.call_args.kwargs["recover"])

    async def test_promised_snapshot_missing_fails_instead_of_starting_fresh(self):
        with self.assertRaisesRegex(FileNotFoundError, "says a Copilot snapshot"):
            await self.run_task(state={"has_snapshot": True})
        QueueHarness.instances[-1].start.assert_not_awaited()

    async def test_legacy_config_and_marker_migrate_before_fresh_fallback(self):
        config_root = self.root / "legacy-config-root"
        legacy = config_root / "session"
        legacy.mkdir(parents=True)
        (legacy / "session-store.db").write_bytes(b"legacy-db")
        (legacy / "session-state").mkdir()
        (legacy / "session-state" / "history").write_text(
            "legacy-history", encoding="utf-8"
        )
        new_store = FakeStateStore()
        legacy_store = FakeStateStore(
            {
                "pending_elicitation": {
                    "elicitation_id": "legacy-question",
                    "message": "Continue?",
                    "schema": {},
                    "decision": None,
                }
            }
        )
        snapshot = SimpleNamespace(
            restore=AsyncMock(return_value=False), save=AsyncMock()
        )
        stream = FakeStream()

        class StateStores:
            @classmethod
            async def get_or_create(cls, *args, **kwargs):
                return new_store

            def __new__(cls, name):
                self.assertEqual(name, "copilot-state-session")
                return legacy_store

        with patch.object(agent, "_CONFIG_ROOT", config_root), patch.object(
            agent, "_SNAPSHOT_STORE", snapshot
        ), patch.object(agent, "CopilotHarness", ImmediateHarness), patch.object(
            agent.streams, "get_or_create", AsyncMock(return_value=stream)
        ), patch.object(
            agent, "FoundryStateStore", StateStores
        ):
            await agent.copilot_agent._fn(self.context())

        hashed = config_root / session_storage_key("session")
        self.assertEqual((hashed / "session-store.db").read_bytes(), b"legacy-db")
        self.assertEqual(
            (hashed / "session-state" / "history").read_text(encoding="utf-8"),
            "legacy-history",
        )
        self.assertTrue((legacy / "session-store.db").exists())
        snapshot.restore.assert_not_awaited()
        harness = ImmediateHarness.instances[-1]
        self.assertTrue(harness.start.call_args.kwargs["recover"])
        self.assertEqual(
            new_store.value["pending_elicitation"]["elicitation_id"],
            "legacy-question",
        )
        legacy_store.aclose.assert_awaited_once()

    async def test_legacy_marker_migrates_after_config_was_already_copied(self):
        config_root = self.root / "partial-migration-root"
        hashed = config_root / session_storage_key("session")
        hashed.mkdir(parents=True)
        (hashed / "session-store.db").write_bytes(b"already-copied")
        new_store = FakeStateStore()
        legacy_store = FakeStateStore(
            {
                "pending_elicitation": {
                    "elicitation_id": "legacy-question",
                    "message": "Continue?",
                    "schema": {},
                    "decision": None,
                }
            }
        )
        snapshot = SimpleNamespace(
            restore=AsyncMock(return_value=False), save=AsyncMock()
        )
        stream = FakeStream()

        class StateStores:
            @classmethod
            async def get_or_create(cls, *args, **kwargs):
                return new_store

            def __new__(cls, name):
                self.assertEqual(name, "copilot-state-session")
                return legacy_store

        with patch.object(agent, "_CONFIG_ROOT", config_root), patch.object(
            agent, "_SNAPSHOT_STORE", snapshot
        ), patch.object(agent, "CopilotHarness", ImmediateHarness), patch.object(
            agent.streams, "get_or_create", AsyncMock(return_value=stream)
        ), patch.object(
            agent, "FoundryStateStore", StateStores
        ):
            await agent.copilot_agent._fn(self.context())

        snapshot.restore.assert_not_awaited()
        self.assertEqual(
            new_store.value["pending_elicitation"]["elicitation_id"],
            "legacy-question",
        )
        self.assertTrue(
            ImmediateHarness.instances[-1].start.call_args.kwargs["recover"]
        )
        legacy_store.aclose.assert_awaited_once()

    async def test_shutdown_defers_without_closing_stream_or_failure_frame(self):
        class StopHarness(QueueHarness):
            async def events(self):
                while not (self.abort.await_count):
                    await asyncio.sleep(0)
                if False:
                    yield None

        shutdown = asyncio.Event()
        shutdown.set()
        context = TaskContext(
            task_id="copilot-session",
            session_id="session",
            input={
                "invocation_id": "invocation",
                "prompt": "initial",
                "session_id": "session",
            },
            recovery_count=1,
            entry_mode="recovered",
            shutdown=shutdown,
        )
        result, stream, _, snapshot = await self.run_task(
            context=context, harness=StopHarness
        )
        self.assertEqual(type(result).__name__, "_ExitForRecovery")
        stream.close.assert_not_awaited()
        self.assertNotIn("turn_failed", [frame["type"] for frame in stream.frames])
        snapshot.save.assert_awaited()

    async def test_explicit_cancel_is_not_resurrected(self):
        class StopHarness(QueueHarness):
            async def events(self):
                while not self.abort.await_count:
                    await asyncio.sleep(0)
                if False:
                    yield None

        context = self.context(cancel=True, shutdown=True)
        agent.mark_client_cancel("session")
        _, stream, _, _ = await self.run_task(context=context, harness=StopHarness)
        context.exit_for_recovery.assert_not_awaited()
        self.assertIn("cancelled", [frame["type"] for frame in stream.frames])
        stream.close.assert_awaited_once()

    async def test_native_input_waits_for_ready_and_is_durable_before_202(self):
        log = []
        QueueHarness.start_gate = asyncio.Event()
        QueueHarness.log = log
        context = self.context()
        with patch.object(agent, "_CONFIG_ROOT", self.root / "config"):
            slot = agent.reserve_harness("session")
        self.assertFalse(slot.ready.done())

        stream = FakeStream()
        state_store = FakeStateStore({}, log)
        snapshot = SimpleNamespace(
            restore=AsyncMock(return_value=False),
            save=AsyncMock(side_effect=lambda *args, **kwargs: log.append("snapshot")),
        )
        with patch.object(agent, "_CONFIG_ROOT", self.root / "config"), patch.object(
            agent, "_SNAPSHOT_STORE", snapshot
        ), patch.object(agent, "CopilotHarness", QueueHarness), patch.object(
            agent.streams, "get_or_create", AsyncMock(return_value=stream)
        ), patch.object(
            agent.FoundryStateStore,
            "get_or_create",
            AsyncMock(return_value=state_store),
        ):
            task = asyncio.create_task(agent.copilot_agent._fn(context))
            submit = asyncio.create_task(
                agent.submit_native_input(
                    "session", "input-1", "follow up", "immediate"
                )
            )
            await asyncio.sleep(0)
            self.assertFalse(submit.done())
            QueueHarness.start_gate.set()
            result = await asyncio.wait_for(submit, 2)
            self.assertEqual(result["status"], "delivered")
            harness = QueueHarness.instances[-1]
            self.assertLess(log.index("state:accepted"), log.index("send:follow up"))
            self.assertLess(log.index("send:follow up"), log.index("snapshot"))
            duplicate = await agent.submit_native_input(
                "session", "input-1", "follow up", "immediate"
            )
            self.assertTrue(duplicate["deduplicated"])
            with self.assertRaisesRegex(ValueError, "different prompt"):
                await agent.submit_native_input(
                    "session", "input-1", "different", "immediate"
                )
            with self.assertRaisesRegex(ValueError, "exceed 512 KiB"):
                await agent.submit_native_input(
                    "session", "too-large", "x" * (513 * 1024), "enqueue"
                )
            self.assertFalse(task.done())
            await harness.queue.put(None)
            await asyncio.wait_for(task, 2)
            self.assertEqual(state_store.value["input_journal"], [])
            self.assertIsNone(
                await agent.submit_native_input(
                    "session", "after-close", "later", "enqueue"
                )
            )

    async def test_idle_queued_during_acceptance_does_not_cover_native_input(self):
        harness = CopilotHarness()
        idle = SimpleNamespace(type=SessionEventType.SESSION_IDLE)

        class Session:
            abort = AsyncMock()

            async def send(self, prompt, *, mode=None):
                return f"message-{prompt}"

        harness.session = Session()
        harness.start = AsyncMock()
        stream = FakeStream()
        snapshot = SimpleNamespace(
            restore=AsyncMock(return_value=False), save=AsyncMock()
        )

        class IdleInjectingStateStore(FakeStateStore):
            def __init__(self):
                super().__init__({})
                self.injected = False

            async def set_item(self, key, value):
                await super().set_item(key, value)
                statuses = [
                    entry.get("status") for entry in value.get("input_journal", [])
                ]
                if statuses == ["accepted"] and not self.injected:
                    self.injected = True
                    harness._on_event(idle)

        state_store = IdleInjectingStateStore()
        with patch.object(
            agent, "_CONFIG_ROOT", self.root / "config-idle"
        ), patch.object(agent, "_SNAPSHOT_STORE", snapshot), patch.object(
            agent, "CopilotHarness", return_value=harness
        ), patch.object(
            agent.streams, "get_or_create", AsyncMock(return_value=stream)
        ), patch.object(
            agent.FoundryStateStore,
            "get_or_create",
            AsyncMock(return_value=state_store),
        ):
            task = asyncio.create_task(agent.copilot_agent._fn(self.context()))
            while not harness.start.await_count:
                await asyncio.sleep(0)
            result = await agent.submit_native_input(
                "session", "input-idle", "follow up", "enqueue"
            )
            self.assertEqual(result["status"], "delivered")
            self.assertFalse(task.done())
            self.assertEqual(
                state_store.value["input_journal"][0]["status"], "delivered"
            )
            harness._on_event(idle)
            await asyncio.wait_for(task, 2)
            self.assertEqual(state_store.value["input_journal"], [])

    async def test_during_close_routes_away_from_the_old_harness(self):
        class ClosingHarness(QueueHarness):
            close_started = None
            close_gate = None

            async def _close(self):
                type(self).close_started.set()
                await type(self).close_gate.wait()
                self.stop_accepting_controls()

        ClosingHarness.close_started = asyncio.Event()
        ClosingHarness.close_gate = asyncio.Event()
        stream = FakeStream()
        state_store = FakeStateStore({})
        snapshot = SimpleNamespace(
            restore=AsyncMock(return_value=False), save=AsyncMock()
        )
        with patch.object(agent, "_CONFIG_ROOT", self.root / "config"), patch.object(
            agent, "_SNAPSHOT_STORE", snapshot
        ), patch.object(agent, "CopilotHarness", ClosingHarness), patch.object(
            agent.streams, "get_or_create", AsyncMock(return_value=stream)
        ), patch.object(
            agent.FoundryStateStore,
            "get_or_create",
            AsyncMock(return_value=state_store),
        ):
            task = asyncio.create_task(agent.copilot_agent._fn(self.context()))
            while not ClosingHarness.instances:
                await asyncio.sleep(0)
            harness = ClosingHarness.instances[-1]
            while not harness.start.await_count:
                await asyncio.sleep(0)
            await harness.queue.put(None)
            await ClosingHarness.close_started.wait()
            self.assertIsNone(
                await agent.submit_native_input(
                    "session", "during-close", "later", "enqueue"
                )
            )
            ClosingHarness.close_gate.set()
            await task

    async def test_normal_completion_leaves_no_stop_watcher_tasks(self):
        await self.run_task()
        await asyncio.sleep(0)
        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and "watch_stop" in repr(task.get_coro())
        ]
        self.assertEqual(pending, [])

    async def test_second_question_during_first_snapshot_keeps_both_ids(self):
        state = {
            "pending_elicitation": {
                "elicitation_id": "question-1",
                "message": "First?",
                "schema": {},
                "decision": None,
            }
        }
        stream = FakeStream()
        state_store = FakeStateStore(state)
        question_2_ack = asyncio.get_running_loop().create_future()
        injected = False

        async def save(_session_id, _config_dir, *, live=False):
            nonlocal injected
            if live and not injected:
                injected = True
                await QueueHarness.instances[-1].queue.put(
                    ElicitationRequest(
                        "question-2",
                        "Second?",
                        {},
                        question_2_ack,
                    )
                )

        snapshot = SimpleNamespace(
            restore=AsyncMock(return_value=False),
            save=AsyncMock(side_effect=save),
        )
        with patch.object(
            agent, "_CONFIG_ROOT", self.root / "config-two-questions"
        ), patch.object(agent, "_SNAPSHOT_STORE", snapshot), patch.object(
            agent, "CopilotHarness", QueueHarness
        ), patch.object(
            agent.streams, "get_or_create", AsyncMock(return_value=stream)
        ), patch.object(
            agent.FoundryStateStore,
            "get_or_create",
            AsyncMock(return_value=state_store),
        ):
            task = asyncio.create_task(agent.copilot_agent._fn(self.context()))
            while not QueueHarness.instances:
                await asyncio.sleep(0)
            harness = QueueHarness.instances[-1]
            while not harness.start.await_count:
                await asyncio.sleep(0)
            result = await agent.submit_elicitation_decision(
                "session", "question-1", "accept", {}
            )
            self.assertEqual(result["status"], "delivered_as_message")
            await asyncio.wait_for(question_2_ack, 2)
            markers = state_store.value["elicitations"]
            self.assertEqual(markers["question-1"]["delivery"], "submitted")
            self.assertIsNone(markers["question-2"]["decision"])
            self.assertEqual(
                state_store.value["pending_elicitation"]["elicitation_id"],
                "question-2",
            )
            await harness.queue.put(None)
            await asyncio.wait_for(task, 2)
            markers = state_store.value["elicitations"]
            self.assertNotIn("question-1", markers)
            self.assertIn("question-2", markers)

    async def test_todo_checkpoint_does_not_confirm_blocked_hitl_response(self):
        state = {
            "pending_elicitation": {
                "elicitation_id": "question-1",
                "message": "Continue?",
                "schema": {},
                "decision": None,
            }
        }
        stream = FakeStream()
        state_store = FakeStateStore(state)
        checkpoint_ack = asyncio.get_running_loop().create_future()
        injected = False

        async def save(_session_id, _config_dir, *, live=False):
            nonlocal injected
            if live and not injected:
                injected = True
                await QueueHarness.instances[-1].queue.put(
                    TodoCheckpointRequest("parallel-tool", checkpoint_ack)
                )

        snapshot = SimpleNamespace(
            restore=AsyncMock(return_value=False),
            save=AsyncMock(side_effect=save),
        )
        with patch.object(
            agent, "_CONFIG_ROOT", self.root / "config-hitl-checkpoint"
        ), patch.object(agent, "_SNAPSHOT_STORE", snapshot), patch.object(
            agent, "CopilotHarness", QueueHarness
        ), patch.object(
            agent.streams, "get_or_create", AsyncMock(return_value=stream)
        ), patch.object(
            agent.FoundryStateStore,
            "get_or_create",
            AsyncMock(return_value=state_store),
        ):
            task = asyncio.create_task(agent.copilot_agent._fn(self.context()))
            while not QueueHarness.instances:
                await asyncio.sleep(0)
            harness = QueueHarness.instances[-1]
            while not harness.start.await_count:
                await asyncio.sleep(0)
            result = await agent.submit_elicitation_decision(
                "session", "question-1", "accept", {}
            )
            self.assertEqual(result["status"], "delivered_as_message")
            await asyncio.wait_for(checkpoint_ack, 2)
            self.assertEqual(
                state_store.value["elicitations"]["question-1"]["delivery"],
                "submitted",
            )
            await harness.queue.put(None)
            await asyncio.wait_for(task, 2)
            self.assertNotIn("question-1", state_store.value["elicitations"])

    async def test_recovery_replays_uncovered_accepted_inputs(self):
        state = {
            "input_journal": [
                {
                    "input_id": "accepted",
                    "prompt": "accepted prompt",
                    "mode": "enqueue",
                    "status": "delivered",
                }
            ]
        }

        async def restore(_session_id, config_dir):
            config_dir.mkdir(parents=True)
            (config_dir / "session-store.db").write_bytes(b"db")
            return True

        _, _, _, _ = await self.run_task(
            context=self.context(entry_mode="recovered"),
            state=state,
            restore_side_effect=restore,
        )
        harness = ImmediateHarness.instances[-1]
        self.assertEqual(
            harness.sent[:2], [("initial", None), ("accepted prompt", "enqueue")]
        )

    async def test_recovered_hitl_decisions_remain_until_idle_and_keep_empty_accept(
        self,
    ):
        for action, content in (
            ("accept", {}),
            ("decline", None),
            ("cancel", None),
        ):
            with self.subTest(action=action):
                agent._HARNESS_SLOTS.clear()
                QueueHarness.instances.clear()
                state = {
                    "pending_elicitation": {
                        "elicitation_id": "elicitation-1",
                        "message": "Continue?",
                        "schema": {},
                        "decision": None,
                    }
                }
                stream = FakeStream()
                state_store = FakeStateStore(state)
                snapshot = SimpleNamespace(
                    restore=AsyncMock(return_value=False), save=AsyncMock()
                )
                with patch.object(
                    agent, "_CONFIG_ROOT", self.root / f"config-{action}"
                ), patch.object(agent, "_SNAPSHOT_STORE", snapshot), patch.object(
                    agent, "CopilotHarness", QueueHarness
                ), patch.object(
                    agent.streams, "get_or_create", AsyncMock(return_value=stream)
                ), patch.object(
                    agent.FoundryStateStore,
                    "get_or_create",
                    AsyncMock(return_value=state_store),
                ):
                    task = asyncio.create_task(agent.copilot_agent._fn(self.context()))
                    while not QueueHarness.instances:
                        await asyncio.sleep(0)
                    harness = QueueHarness.instances[-1]
                    while not harness.start.await_count:
                        await asyncio.sleep(0)
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        await agent.submit_elicitation_decision(
                            "session", "stale", action, content
                        )
                    result = await agent.submit_elicitation_decision(
                        "session", "elicitation-1", action, content
                    )
                    self.assertIn(
                        result["status"], {"answered", "delivered_as_message"}
                    )
                    payload = json.loads(harness.sent[-1][0].split("] ", 1)[1])
                    self.assertEqual(payload["action"], action)
                    if action == "accept":
                        self.assertEqual(payload["content"], {})
                    self.assertEqual(
                        state_store.value["pending_elicitation"]["delivery"],
                        "submitted",
                    )
                    await harness.queue.put(None)
                    await task
                    self.assertIsNone(state_store.value["pending_elicitation"])

    def test_crash_hook_is_disabled_for_recovered_entry(self):
        with patch.object(agent, "_SIMULATE_CRASH_AFTER_EVENTS", 1), patch.object(
            agent.os, "_exit"
        ) as exit_mock:
            agent._maybe_crash(5, enabled=False)
            exit_mock.assert_not_called()
            agent._maybe_crash(1, enabled=True)
            exit_mock.assert_called_once_with(1)


class StartupEnvironmentTests(unittest.TestCase):
    def test_dotenv_is_loaded_before_environment_dependent_local_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "FOUNDRY_PROJECT_ENDPOINT=https://example.test\n"
                "SIMULATE_CRASH_AFTER_EVENTS=7\n",
                encoding="utf-8",
            )
            script = (
                "import sys;"
                f"sys.path.insert(0, {str(SRC)!r});"
                "import main,agent;"
                "print(agent._SNAPSHOT_STORE._use_state_store,"
                "agent._SIMULATE_CRASH_AFTER_EVENTS)"
            )
            env = os.environ.copy()
            env.pop("FOUNDRY_PROJECT_ENDPOINT", None)
            env.pop("SIMULATE_CRASH_AFTER_EVENTS", None)
            env["AGENTSERVER_STATE_ROOT"] = str(root / "state")
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        self.assertIn("True 7", result.stdout)


class HttpValidationTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def request(body, *, session_id, invocation_id="invocation"):
        sent = False

        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {
                "type": "http.request",
                "body": json.dumps(body).encode(),
                "more_body": False,
            }

        request = main.Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/invocations",
                "query_string": b"",
                "headers": [],
            },
            receive,
        )
        request.state.session_id = session_id
        request.state.invocation_id = invocation_id
        request.state.call_id = "call"
        return request

    async def test_invalid_http_session_id_returns_standard_400(self):
        response = await main.handle_invoke(
            self.request({"input": "hello"}, session_id="../escape")
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.body)["error"], "invalid_request")

    async def test_no_live_harness_falls_back_to_framework_durable_turn(self):
        request = self.request(
            {"input": "hello", "mode": "immediate", "input_id": "stable"},
            session_id="session",
        )
        stream = SimpleNamespace()
        with patch.object(
            main, "submit_native_input", AsyncMock(return_value=None)
        ), patch.object(
            main.streams, "get_or_create", AsyncMock(return_value=stream)
        ), patch.object(
            main.copilot_agent, "start", AsyncMock()
        ) as start, patch.object(
            main, "reserve_harness"
        ):
            response = await main.handle_invoke(request)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(json.loads(response.body)["action"], "start")
        start.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
