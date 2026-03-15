"""Async in-process event bus backed by asyncio.Queue.

Consumers are expected to be idempotent; the bus provides at-least-once
delivery within a single process lifetime.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Awaitable, Callable, Type, TypeVar

from exo.models import PipelineEvent

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=PipelineEvent)
Handler = Callable[[PipelineEvent], Awaitable[None]]


class EventBus:
    """Simple async pub/sub event bus.

    Usage::

        bus = EventBus()

        @bus.subscribe(FeatureUpdated)
        async def on_feature_updated(event: FeatureUpdated) -> None:
            ...

        await bus.publish(FeatureUpdated(record=rec))
        await bus.drain()
    """

    def __init__(self, queue_size: int = 0) -> None:
        # event_type → list of async handlers
        self._handlers: dict[type, list[Handler]] = defaultdict(list)
        self._queue: asyncio.Queue[PipelineEvent] = asyncio.Queue(maxsize=queue_size)
        self._running = False
        self._worker_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def subscribe(self, event_type: Type[T]) -> Callable[[Handler], Handler]:
        """Decorator to register an async handler for *event_type*."""

        def decorator(fn: Handler) -> Handler:
            self._handlers[event_type].append(fn)
            return fn

        return decorator

    def register(self, event_type: Type[T], handler: Handler) -> None:
        """Programmatically register a handler."""
        self._handlers[event_type].append(handler)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, event: PipelineEvent) -> None:
        """Enqueue *event* for delivery to all registered handlers."""
        await self._queue.put(event)

    def publish_sync(self, event: PipelineEvent) -> None:
        """Thread-safe publish from a non-async context (best-effort)."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("EventBus queue full; dropping event %s", event.event_id)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    async def _dispatch(self, event: PipelineEvent) -> None:
        handlers = self._handlers.get(type(event), [])
        if not handlers:
            logger.debug("No handlers for event type %s", type(event).__name__)
            return
        results = await asyncio.gather(
            *[h(event) for h in handlers], return_exceptions=True
        )
        for r in results:
            if isinstance(r, Exception):
                logger.error("Handler raised exception for %s: %s", type(event).__name__, r)

    async def _worker(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                await self._dispatch(event)
            finally:
                self._queue.task_done()

    async def start(self) -> None:
        """Start the background worker task."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker(), name="event-bus-worker")

    async def stop(self) -> None:
        """Gracefully stop the worker after draining the queue."""
        self._running = False
        if self._worker_task:
            await self._queue.join()
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def drain(self) -> None:
        """Wait until all currently queued events have been dispatched."""
        await self._queue.join()


# Module-level singleton
_bus: EventBus | None = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
