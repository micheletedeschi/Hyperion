"""Schedule periodic dataset updates using APScheduler."""

import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from hyperion3.config.base_config import HyperionV2Config
from hyperion3.data.downloader import DataDownloader


def _parse_seconds(freq: str) -> int:
    """Parse a frequency string like '1h' or '30m' into seconds."""
    freq = freq.strip().lower()
    if freq.endswith("h"):
        return int(freq[:-1]) * 3600
    if freq.endswith("m"):
        return int(freq[:-1]) * 60
    if freq.endswith("s"):
        return int(freq[:-1])
    if freq.endswith("d"):
        return int(freq[:-1]) * 86400
    return int(freq)


async def download_all(config: HyperionV2Config) -> None:
    """Download data for all configured symbols."""
    async with DataDownloader(config) as downloader:
        for symbol in config.data.symbols:
            await downloader.download_symbol(symbol)


def start_scheduler(config: HyperionV2Config) -> AsyncIOScheduler:
    """Start the scheduler and return it."""
    try:
        asyncio.get_running_loop()
        scheduler = AsyncIOScheduler()
    except RuntimeError:
        scheduler = BackgroundScheduler()
    interval = _parse_seconds(config.data.update_frequency)
    scheduler.add_job(
        lambda: asyncio.create_task(download_all(config)),
        trigger="interval",
        seconds=interval,
    )
    scheduler.start()
    return scheduler


if __name__ == "__main__":
    cfg = HyperionV2Config()
    start_scheduler(cfg)
    asyncio.get_event_loop().run_forever()
