import os
from dataclasses import dataclass, field
from loguru import logger

import httpx

# Use very small timeouts because we expect to be operating locally.
# (and if the server isn't online, we want to drop requests immediately instead of lagging/retrying forever)
httpxtimeout = httpx.Timeout(0.333, connect=0.333, write=0.333, read=None)

ICLI_AWWDIO_URL = os.getenv("ICLI_AWWDIO_URL", None)


@dataclass
class AwwdioClient:
    url: str | None = ICLI_AWWDIO_URL

    client: httpx.AsyncClient = field(
        default_factory=lambda: httpx.AsyncClient(timeout=httpxtimeout)
    )

    async def say(
        self, voice: str = "Alex", say: str = "Hello World", speed: int = 250
    ) -> None:
        try:
            await self.client.get(
                f"{self.url}/say", params=dict(voice=voice, say=say, speed=speed)
            )
        except:
            logger.warning("Speaking failed for: {}", say)

    async def sound(self, sound: str = "blip") -> None:
        try:
            await self.client.get(f"{self.url}/play", params=dict(sound=sound))
        except:
            logger.warning("Sound failed to send!")
