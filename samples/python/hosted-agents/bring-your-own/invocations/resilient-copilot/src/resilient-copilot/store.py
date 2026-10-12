"""Durable snapshot store for the GitHub Copilot session workspace.

The Copilot CLI persists each session — conversation history, planning state,
and per-turn checkpoints — to a local ``config_dir`` (a SQLite ``session-store.db``
plus a ``session-state/<id>/`` tree). ``resume_session`` reads that directory.
On a hosted container **replacement** the local disk is gone, so ``resume_session``
alone is not durable: the conversation would be lost.

This store makes the Copilot session **container-durable** by snapshotting the
per-session ``config_dir`` into the **Foundry StateStore**
(:class:`FoundryStateStore`) on todo/plan changes and at each turn boundary,
and restoring it on crash
recovery *before* ``resume_session`` runs — the same durable, platform-managed
store the resilient task itself uses. When no ``FOUNDRY_PROJECT_ENDPOINT`` is
configured (the offline demo), it falls back to an atomic local-file copy so the
sample still runs with no credentials.

Mirrors the two-tier pattern from ``resilient-research``: durable application
state lives in an explicit ``FoundryStateStore`` — small markers hold only a
tiny *reference* (a "snapshot exists" flag); the heavy bytes live here.

The snapshot is a gzip'd tar of the per-session ``config_dir`` tree, base64'd
into a StateStore JSON item. Todo/plan changes also trigger snapshots during
a turn. Live snapshots use SQLite's online backup API, not a raw copy of a
database and its active WAL. Each database is consistent independently; the
archive is not a transaction across all session files. Stale lock files are
excluded so a restored workspace opens cleanly.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import closing
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sqlite3
import tarfile
import tempfile
import time
import zlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

logger = logging.getLogger(__name__)

# Version the store name because get_or_create cannot update the immutable TTL
# of the previously published one-hour store. Restore still checks the legacy
# store so existing snapshots are not abandoned.
_STORE_NAME = "copilot-session-snapshots-v2"
_LEGACY_STORE_NAME = "copilot-session-snapshots"
_ITEM_TTL_SECONDS = 30 * 24 * 60 * 60
_MAX_STATE_VALUE_BYTES = 5 * 1024 * 1024
_STATE_VALUE_JSON_OVERHEAD = len(
    json.dumps({"tar_gz_b64": ""}, separators=(",", ":")).encode("utf-8")
)
_MAX_BASE64_CHARS = ((_MAX_STATE_VALUE_BYTES - _STATE_VALUE_JSON_OVERHEAD) // 4) * 4
_MAX_ARCHIVE_BYTES = (_MAX_BASE64_CHARS // 4) * 3
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,120}$")
_INVOCATION_STORE_NAME = "copilot-invocations"
_INVOCATION_TTL_SECONDS = 30 * 24 * 60 * 60


class SnapshotTooLargeError(ValueError):
    """The serialized StateStore value exceeds the service's hard maximum."""


def _snapshot_too_large_error() -> SnapshotTooLargeError:
    return SnapshotTooLargeError(
        "Copilot session snapshot exceeds Foundry StateStore's 5 MiB serialized "
        "JSON value limit. Reduce the session workspace before retrying. "
        "Deployments without StateStore value offload may also reject values "
        "above 1 MiB."
    )


class _SizeLimitedBytesIO(io.BytesIO):
    """Bytes buffer that fails before its retained compressed output exceeds a cap."""

    def __init__(self, max_bytes: int) -> None:
        super().__init__()
        self._max_bytes = max_bytes
        self._size = 0

    def write(self, data: bytes) -> int:
        projected_size = max(self._size, self.tell() + len(data))
        if projected_size > self._max_bytes:
            raise _snapshot_too_large_error()
        written = super().write(data)
        self._size = projected_size
        return written


class _GzipBuffer:
    """Streaming gzip writer backed by an optionally size-limited buffer."""

    def __init__(self, max_bytes: int | None) -> None:
        self._buffer = (
            _SizeLimitedBytesIO(max_bytes) if max_bytes is not None else io.BytesIO()
        )
        self._compressor = zlib.compressobj(level=9, wbits=31)
        self._position = 0

    def write(self, data: bytes) -> int:
        compressed = self._compressor.compress(data)
        if compressed:
            self._buffer.write(compressed)
        self._position += len(data)
        return len(data)

    def tell(self) -> int:
        return self._position

    def finish(self) -> None:
        compressed = self._compressor.flush()
        if compressed:
            self._buffer.write(compressed)

    def getvalue(self) -> bytes:
        return self._buffer.getvalue()


def validate_session_id(session_id: str) -> str:
    """Validate a Foundry session id before using it in any boundary API."""
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(
            "session_id must be 1-120 ASCII letters, digits, underscores, or hyphens"
        )
    return session_id


def session_storage_key(session_id: str) -> str:
    """Return a stable, bounded filesystem key for an external session id."""
    validate_session_id(session_id)
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _is_lockfile(name: str) -> bool:
    """Stale per-process lock files must not travel into a restored workspace."""
    base = os.path.basename(name)
    return base.endswith(".lock") or ".lock" in base


def todo_fingerprint(workspace: Path) -> str | None:
    """Identify committed todo/dependency rows and the session's plan file."""
    state: dict[str, Any] = {}
    db_path = workspace / "session.db"
    if db_path.exists():
        with closing(
            sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=5)
        ) as db:
            db.execute("BEGIN")
            for table in ("todos", "todo_deps"):
                exists = db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if exists:
                    cursor = db.execute(f"SELECT * FROM {table}")
                    columns = [column[0] for column in cursor.description]
                    rows = sorted(
                        (list(row) for row in cursor.fetchall()),
                        key=lambda row: json.dumps(row, default=str),
                    )
                    state[table] = {"columns": columns, "rows": rows}
    plan = workspace / "plan.md"
    if plan.exists():
        state["plan"] = hashlib.sha256(plan.read_bytes()).hexdigest()
    if not state:
        return None
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _sqlite_backup(src: Path, dst: Path) -> None:
    deadline = time.monotonic() + 10

    def progress(status: int, remaining: int, total: int) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out snapshotting SQLite database {src.name}")

    with closing(
        sqlite3.connect(src.as_uri() + "?mode=ro", uri=True, timeout=5)
    ) as source, closing(sqlite3.connect(dst)) as target:
        source.backup(target, pages=128, progress=progress, sleep=0.01)


def _complete_jsonl_prefix_size(path: Path) -> int:
    """Return the byte length through the final complete JSONL record."""
    size = path.stat().st_size
    with path.open("rb") as source:
        position = size
        while position:
            chunk_size = min(position, 64 * 1024)
            position -= chunk_size
            source.seek(position)
            chunk = source.read(chunk_size)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                return position + newline + 1
    return 0


def _pack_dir(
    src: Path, *, live: bool = False, max_archive_bytes: int | None = None
) -> bytes:
    """Return a gzip'd tar of *src*'s contents (excluding stale lock files)."""
    output = _GzipBuffer(max_archive_bytes)
    with tempfile.TemporaryDirectory(prefix="copilot-snapshot-") as staging:
        stage = Path(staging)
        with tarfile.open(fileobj=output, mode="w|") as tar:
            for item in sorted(src.rglob("*")):
                if not item.is_file() or _is_lockfile(item.name):
                    continue
                relative = item.relative_to(src)
                if live and item.name.endswith(("-wal", "-shm", "-journal")):
                    continue
                if live and item.suffix in (".db", ".sqlite", ".sqlite3"):
                    backup = stage / relative
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    _sqlite_backup(item, backup)
                    tar.add(backup, arcname=relative.as_posix())
                elif live and item.suffix == ".jsonl":
                    info = tarfile.TarInfo(relative.as_posix())
                    info.size = _complete_jsonl_prefix_size(item)
                    with item.open("rb") as source:
                        tar.addfile(info, source)
                else:
                    tar.add(item, arcname=relative.as_posix())
    output.finish()
    return output.getvalue()


def _unpack_dir(blob: bytes, dst: Path) -> None:
    """Restore a gzip'd tar *blob* into *dst* (created fresh)."""
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            original_name = member.name
            name = original_name.replace("\\", "/")
            path = PurePosixPath(name)
            if (
                not name
                or path.is_absolute()
                or ".." in path.parts
                or PureWindowsPath(original_name).drive
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe snapshot archive member: {original_name!r}")
            member.name = name

        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=dst.parent, prefix=f"{dst.name}-restore-"
        ) as staging_name:
            staging = Path(staging_name)
            tar.extractall(staging, members=members, filter="data")
            if dst.exists():
                shutil.rmtree(dst)
            staging.replace(dst)


class SessionSnapshotStore:
    """Durable session-id -> workspace-snapshot store (StateStore, file fallback)."""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir
        self._base.mkdir(parents=True, exist_ok=True)
        # Use the Foundry StateStore when an endpoint is available; otherwise
        # fall back to the local file store for the offline demo.
        self._use_state_store = bool(os.environ.get("FOUNDRY_PROJECT_ENDPOINT"))
        self._store: Any = None
        self._legacy_store: Any = None

    async def _state_store(self) -> Any:
        """Lazily resolve (or create) the Foundry StateStore for this agent."""
        if self._store is None:
            from azure.ai.agentserver.core.storage import (  # noqa: PLC0415
                FoundryStateStore,
            )

            self._store = await FoundryStateStore.get_or_create(
                _STORE_NAME,
                item_ttl_seconds=_ITEM_TTL_SECONDS,
            )
        return self._store

    async def _legacy_state_store(self) -> Any:
        """Bind the old store without creating it, solely for migration reads."""
        if self._legacy_store is None:
            from azure.ai.agentserver.core.storage import (  # noqa: PLC0415
                FoundryStateStore,
            )

            self._legacy_store = FoundryStateStore(_LEGACY_STORE_NAME)
        return self._legacy_store

    async def save(
        self, session_id: str, config_dir: Path, *, live: bool = False
    ) -> None:
        """Snapshot the per-session *config_dir* tree under *session_id*.

        Use ``live=True`` while the CLI is running to back up SQLite databases
        and exclude incomplete JSONL records. Otherwise close the CLI first.
        """
        validate_session_id(session_id)
        if not config_dir.exists():
            raise FileNotFoundError(
                f"Copilot session directory not found: {config_dir}"
            )
        blob = await asyncio.to_thread(
            _pack_dir,
            config_dir,
            live=live,
            max_archive_bytes=_MAX_ARCHIVE_BYTES if self._use_state_store else None,
        )
        if self._use_state_store:
            store = await self._state_store()
            # StateStore item values are JSON objects, so base64 the archive.
            encoded = base64.b64encode(blob).decode("ascii")
            value = {"tar_gz_b64": encoded}
            serialized_size = _STATE_VALUE_JSON_OVERHEAD + len(encoded)
            if serialized_size > _MAX_STATE_VALUE_BYTES:
                raise _snapshot_too_large_error()
            await store.set_item(session_id, value)
            logger.info(
                "Saved Copilot session snapshot to StateStore "
                "(%d archive bytes, %d serialized bytes) for %s",
                len(blob),
                serialized_size,
                session_id,
            )
            return

        target = self._path(session_id)
        fd, tmp = tempfile.mkstemp(
            dir=str(self._base),
            prefix=f"{session_storage_key(session_id)}_",
            suffix=".tgz",
        )
        try:
            with open(fd, "wb") as fh:
                fh.write(blob)
            Path(tmp).replace(target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    async def restore(self, session_id: str, config_dir: Path) -> bool:
        """Restore the snapshot for *session_id* into *config_dir*.

        Returns True if a snapshot was found and restored, else False (a fresh
        session should be created).
        """
        validate_session_id(session_id)
        blob: bytes | None = None
        if self._use_state_store:
            store = await self._state_store()
            item = await store.get_item(session_id)
            if item is None:
                from azure.ai.agentserver.core.storage import (  # noqa: PLC0415
                    FoundryStorageNotFoundError,
                )

                legacy = await self._legacy_state_store()
                try:
                    item = await legacy.get_item(session_id)
                except FoundryStorageNotFoundError:
                    item = None
            if item is not None:
                encoded = (item.value or {}).get("tar_gz_b64")
                if encoded:
                    blob = base64.b64decode(encoded)
        else:
            path = self._path(session_id)
            if not path.exists():
                path = self._legacy_path(session_id)
            if path.exists():
                blob = path.read_bytes()

        if blob is None:
            return False
        _unpack_dir(blob, config_dir)
        logger.info(
            "Restored Copilot session snapshot (%d bytes) for %s", len(blob), session_id
        )
        return True

    async def delete(self, session_id: str) -> None:
        """Remove the snapshot for *session_id* if present; no-op otherwise."""
        validate_session_id(session_id)
        if self._use_state_store:
            from azure.ai.agentserver.core.storage import (  # noqa: PLC0415
                FoundryStorageNotFoundError,
            )

            store = await self._state_store()
            try:
                await store.delete_item(session_id)
            except FoundryStorageNotFoundError:
                pass
            return

        path = self._path(session_id)
        if path.exists():
            path.unlink()
        legacy = self._legacy_path(session_id)
        if legacy.exists():
            legacy.unlink()

    def _path(self, session_id: str) -> Path:
        return self._base / f"{session_storage_key(session_id)}.tgz"

    def _legacy_path(self, session_id: str) -> Path:
        """Return the pre-hash filename for migration of already-valid ids."""
        return self._base / f"{session_id}.tgz"

    async def aclose(self) -> None:
        """Close any StateStore clients created by this snapshot store."""
        for store in (self._store, self._legacy_store):
            if store is not None:
                await store.aclose()
        self._store = None
        self._legacy_store = None


class InvocationStatusStore:
    """Durable invocation-id -> polling status store."""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir
        self._base.mkdir(parents=True, exist_ok=True)
        self._use_state_store = bool(os.environ.get("FOUNDRY_PROJECT_ENDPOINT"))
        self._store: Any = None

    async def _state_store(self) -> Any:
        if self._store is None:
            from azure.ai.agentserver.core.storage import (  # noqa: PLC0415
                FoundryStateStore,
            )

            self._store = await FoundryStateStore.get_or_create(
                _INVOCATION_STORE_NAME,
                item_ttl_seconds=_INVOCATION_TTL_SECONDS,
            )
        return self._store

    async def get(self, invocation_id: str) -> dict[str, Any] | None:
        if self._use_state_store:
            item = await (await self._state_store()).get_item(invocation_id)
            return (
                dict(item.value)
                if item is not None and isinstance(item.value, dict)
                else None
            )
        path = self._path(invocation_id)
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, dict) else None

    async def put(
        self,
        invocation_id: str,
        *,
        task_id: str,
        framework_input_id: str,
        status: str,
        task_status: str,
        payload: Any = None,
    ) -> None:
        value = {
            "task_id": task_id,
            "invocation_id": invocation_id,
            "framework_input_id": framework_input_id,
            "status": status,
            "task_status": task_status,
            "payload": payload,
        }
        if self._use_state_store:
            await (await self._state_store()).set_item(invocation_id, value)
            return
        await self._write_local(invocation_id, value)

    async def _write_local(self, invocation_id: str, value: dict[str, Any]) -> None:
        target = self._path(invocation_id)
        fd, tmp = tempfile.mkstemp(
            dir=str(self._base),
            prefix=f"{hashlib.sha256(invocation_id.encode()).hexdigest()}_",
            suffix=".tmp",
        )
        try:
            with open(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle)
            Path(tmp).replace(target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _path(self, invocation_id: str) -> Path:
        key = hashlib.sha256(invocation_id.encode("utf-8")).hexdigest()
        return self._base / f"{key}.json"
