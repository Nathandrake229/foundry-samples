import asyncio
import base64
from contextlib import closing
import importlib
import io
import os
from pathlib import Path
import sqlite3
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


SRC = Path(__file__).resolve().parents[1] / "src" / "resilient-copilot"
sys.path.insert(0, str(SRC))

from copilot_session import CopilotHarness, TodoCheckpointRequest
from store import SessionSnapshotStore, SnapshotTooLargeError, todo_fingerprint


def create_todos(workspace: Path) -> sqlite3.Connection:
    workspace.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(workspace / "session.db")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE todos (id TEXT PRIMARY KEY, title TEXT, status TEXT)")
    db.execute("CREATE TABLE todo_deps (todo_id TEXT, depends_on TEXT)")
    db.commit()
    return db


class TodoSnapshotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "config"
        self.workspace = self.config / "session-state" / "test"
        self.db = create_todos(self.workspace)
        self.addCleanup(self.db.close)
        self.store = SessionSnapshotStore(self.root / "snapshots")
        self.store._use_state_store = False

    def change_todo(self, status: str):
        self.db.execute(
            "INSERT INTO todos VALUES ('a', 'First step', ?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status",
            (status,),
        )
        self.db.commit()

    def test_fingerprint_tracks_todos_dependencies_plan_and_deletion(self):
        empty = todo_fingerprint(self.workspace)
        self.change_todo("pending")
        pending = todo_fingerprint(self.workspace)
        self.assertNotEqual(empty, pending)
        self.db.execute("SELECT * FROM todos").fetchall()
        self.assertEqual(pending, todo_fingerprint(self.workspace))
        self.change_todo("done")
        done = todo_fingerprint(self.workspace)
        self.assertNotEqual(pending, done)
        self.db.execute("INSERT INTO todo_deps VALUES ('b', 'a')")
        self.db.commit()
        deps = todo_fingerprint(self.workspace)
        self.assertNotEqual(done, deps)
        (self.workspace / "plan.md").write_text("A then B", encoding="utf-8")
        plan = todo_fingerprint(self.workspace)
        self.assertNotEqual(deps, plan)
        self.db.execute("DELETE FROM todos")
        self.db.commit()
        self.assertNotEqual(plan, todo_fingerprint(self.workspace))

    async def test_live_snapshot_restores_committed_wal_todos_to_new_directory(self):
        self.change_todo("done")
        self.db.execute(
            "INSERT INTO todos VALUES ('uncommitted', 'Not saved', 'pending')"
        )
        (self.workspace / "plan.md").write_text("A then B", encoding="utf-8")
        (self.workspace / "inuse.lock").write_text("stale", encoding="utf-8")
        (self.workspace / "events.jsonl").write_bytes(b'{"type":"saved"}\n{"partial":')
        cloud = SimpleNamespace(set_item=AsyncMock(), get_item=AsyncMock())
        self.store._use_state_store = True
        self.store._store = cloud

        await self.store.save("test", self.config, live=True)
        session_id, value = cloud.set_item.call_args.args
        self.assertEqual(session_id, "test")
        cloud.get_item.return_value = SimpleNamespace(value=value)
        with tarfile.open(
            fileobj=io.BytesIO(base64.b64decode(value["tar_gz_b64"])), mode="r:gz"
        ) as archive:
            names = archive.getnames()
        self.assertFalse(
            any(name.endswith((".lock", "-wal", "-shm")) for name in names)
        )

        restored = self.root / "replacement"
        self.assertTrue(await self.store.restore("test", restored))
        workspace = restored / "session-state" / "test"
        with closing(sqlite3.connect(workspace / "session.db")) as restored_db:
            self.assertEqual(
                restored_db.execute("PRAGMA integrity_check").fetchone()[0], "ok"
            )
            self.assertEqual(
                restored_db.execute("SELECT id, status FROM todos").fetchall(),
                [("a", "done")],
            )
        self.assertEqual((workspace / "plan.md").read_text(), "A then B")
        self.assertEqual(
            (workspace / "events.jsonl").read_bytes(), b'{"type":"saved"}\n'
        )

    async def test_latest_todo_checkpoint_replaces_previous_snapshot(self):
        for status in ("pending", "in_progress", "done"):
            self.change_todo(status)
            await self.store.save("test", self.config, live=True)
        self.assertTrue(await self.store.restore("test", self.root / "replacement"))
        with closing(
            sqlite3.connect(
                self.root / "replacement" / "session-state" / "test" / "session.db"
            )
        ) as restored_db:
            self.assertEqual(
                restored_db.execute("SELECT status FROM todos").fetchone()[0], "done"
            )

    async def test_write_failure_is_not_reported_as_saved(self):
        self.change_todo("pending")
        self.store._use_state_store = True
        self.store._store = SimpleNamespace(
            set_item=AsyncMock(side_effect=OSError("write failed"))
        )
        with self.assertRaisesRegex(OSError, "write failed"):
            await self.store.save("test", self.config, live=True)

    async def test_missing_session_directory_fails_explicitly(self):
        with self.assertRaises(FileNotFoundError):
            await self.store.save("test", self.root / "missing", live=True)

    async def test_hook_waits_for_checkpoint_and_skips_unchanged_todos(self):
        harness = CopilotHarness()
        harness._workspace = self.workspace
        harness._todo_fingerprint = todo_fingerprint(self.workspace)
        for status in ("pending", "in_progress", "done"):
            self.change_todo(status)
            hook = asyncio.create_task(
                harness._post_tool_use({}, {"session_id": "test"})
            )
            request = await asyncio.wait_for(harness._queue.get(), 2)
            self.assertIsInstance(request, TodoCheckpointRequest)
            self.assertFalse(hook.done())
            await self.store.save("test", self.config, live=True)
            request.acknowledged.set_result(None)
            await asyncio.wait_for(hook, 2)
            await harness._post_tool_use({}, {"session_id": "test"})
            self.assertTrue(harness._queue.empty())

    async def test_failed_checkpoint_does_not_advance_fingerprint(self):
        harness = CopilotHarness()
        harness._workspace = self.workspace
        original = todo_fingerprint(self.workspace)
        harness._todo_fingerprint = original
        self.change_todo("pending")
        hook = asyncio.create_task(harness._post_tool_use({}, {}))
        request = await asyncio.wait_for(harness._queue.get(), 2)
        request.acknowledged.set_exception(OSError("checkpoint failed"))
        with self.assertRaisesRegex(OSError, "checkpoint failed"):
            await hook
        self.assertEqual(harness._todo_fingerprint, original)

    async def test_close_unblocks_pending_hook(self):
        harness = CopilotHarness()
        harness._workspace = self.workspace
        harness._todo_fingerprint = todo_fingerprint(self.workspace)
        self.change_todo("pending")
        hook = asyncio.create_task(harness._post_tool_use({}, {}))
        await asyncio.wait_for(harness._queue.get(), 2)
        await harness.close()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(hook, 2)

    async def test_create_and_resume_register_checkpoint_hook(self):
        for recovered in (False, True):
            harness = CopilotHarness()
            session = SimpleNamespace(
                workspace_path=self.workspace, on=lambda cb: lambda: None
            )
            client = SimpleNamespace(
                start=AsyncMock(),
                create_session=AsyncMock(return_value=session),
                resume_session=AsyncMock(return_value=session),
                stop=AsyncMock(),
            )
            with patch(
                "copilot_session.CopilotClient", return_value=client
            ), patch.dict(
                os.environ, {"GITHUB_TOKEN": "unused-test-token"}
            ), patch.object(
                CopilotHarness, "_byok_provider", return_value=(None, None)
            ):
                await harness.start(
                    "test", recover=recovered, config_directory=str(self.config)
                )
            call = client.resume_session if recovered else client.create_session
            self.assertEqual(
                call.call_args.kwargs["hooks"]["on_post_tool_use"],
                harness._post_tool_use,
            )
            await harness.close()


class TaskCheckpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        with patch.dict(os.environ, {"AGENTSERVER_STATE_ROOT": self.tmp.name}):
            self.agent = importlib.import_module("agent")

    async def exercise_task(self, *, checkpoint_failure=False, final_failure=False):
        acknowledgement = asyncio.get_running_loop().create_future()
        request = TodoCheckpointRequest("todo-hash", acknowledgement)

        class Harness:
            has_backend = staticmethod(lambda: True)
            start = AsyncMock()
            send = AsyncMock()
            abort = AsyncMock()
            close = AsyncMock()
            stop_accepting_controls = lambda self: None
            bind_recovered_elicitations = lambda self, markers: None

            async def events(self):
                yield request
                await acknowledgement

        class Stream:
            def __init__(self):
                self.frames = []

            async def last_cursor(self):
                return 0

            async def emit(self, frame):
                self.frames.append(frame)

            close = AsyncMock()

        stream = Stream()
        state_store = SimpleNamespace(
            get_item=AsyncMock(return_value=None),
            set_item=AsyncMock(),
            aclose=AsyncMock(),
        )
        snapshot = SimpleNamespace(
            save=AsyncMock(), restore=AsyncMock(return_value=False)
        )
        expected_exception = None
        if checkpoint_failure:
            failure = (
                checkpoint_failure
                if isinstance(checkpoint_failure, BaseException)
                else OSError("live snapshot failed")
            )
            snapshot.save.side_effect = failure
            expected_exception = type(failure)
        elif final_failure:
            snapshot.save.side_effect = [None, OSError("final snapshot failed")]
            expected_exception = OSError
        context = SimpleNamespace(
            input={"invocation_id": "test-inv", "prompt": "test", "session_id": "test"},
            entry_mode="fresh",
            recovery_count=0,
            cancel=asyncio.Event(),
            shutdown=asyncio.Event(),
        )
        with patch.object(self.agent, "CopilotHarness", Harness), patch.object(
            self.agent, "_SNAPSHOT_STORE", snapshot
        ), patch.object(
            self.agent, "_CONFIG_ROOT", self.root / "task-config"
        ), patch.object(
            self.agent.streams, "get_or_create", AsyncMock(return_value=stream)
        ), patch.object(
            self.agent.FoundryStateStore,
            "get_or_create",
            AsyncMock(return_value=state_store),
        ), patch.object(
            self.agent, "_get_legacy_state_item", AsyncMock(return_value=None)
        ):
            handler = self.agent.copilot_agent._fn
            if expected_exception is not None:
                with self.assertRaises(expected_exception):
                    await handler(context)
            else:
                await handler(context)
        if acknowledgement.done() and not acknowledgement.cancelled():
            acknowledgement.exception()
        return snapshot, stream, state_store

    async def test_task_persists_live_checkpoint_then_completes_same_turn(self):
        snapshot, stream, state_store = await self.exercise_task()
        self.assertEqual(snapshot.save.await_count, 2)
        self.assertTrue(snapshot.save.call_args_list[0].kwargs["live"])
        frames = [frame["type"] for frame in stream.frames]
        self.assertEqual(frames, ["turn_start", "todo_checkpoint", "turn_complete"])
        self.assertTrue(state_store.set_item.call_args.args[1]["has_snapshot"])

    async def test_failed_live_checkpoint_fails_turn_without_completion(self):
        _, stream, state_store = await self.exercise_task(checkpoint_failure=True)
        self.assertEqual(
            [frame["type"] for frame in stream.frames], ["turn_start", "turn_failed"]
        )
        for call in state_store.set_item.call_args_list:
            self.assertNotIn("has_snapshot", call.args[1])

    async def test_oversized_checkpoint_does_not_set_snapshot_marker(self):
        failure = SnapshotTooLargeError("5 MiB serialized JSON limit")
        _, stream, state_store = await self.exercise_task(checkpoint_failure=failure)
        self.assertEqual(
            [frame["type"] for frame in stream.frames], ["turn_start", "turn_failed"]
        )
        for call in state_store.set_item.call_args_list:
            self.assertNotIn("has_snapshot", call.args[1])

    async def test_failed_final_snapshot_does_not_emit_turn_complete(self):
        _, stream, _ = await self.exercise_task(final_failure=True)
        self.assertEqual(
            [frame["type"] for frame in stream.frames],
            ["turn_start", "todo_checkpoint", "turn_failed"],
        )


if __name__ == "__main__":
    unittest.main()
