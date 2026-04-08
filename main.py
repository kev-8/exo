"""exo pipeline entry point.

Starts the full ingestion and risk index scheduler. Runs until interrupted.

Usage:
    python main.py
"""

import asyncio
import logging
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger("exo.main")


async def main() -> None:
    from exo.scheduler import ExoScheduler

    scheduler = ExoScheduler()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await scheduler.start()
    logger.info("exo pipeline running — press Ctrl+C to stop")

    await stop_event.wait()

    logger.info("Shutting down...")
    await scheduler.stop()
    logger.info("exo pipeline stopped")


if __name__ == "__main__":
    asyncio.run(main())
