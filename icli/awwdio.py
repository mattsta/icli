import os
import time
from dataclasses import dataclass, field
from loguru import logger

import httpx

# Use very small timeouts because we expect to be operating locally.
# (and if the server isn't online, we want to drop requests immediately instead of lagging/retrying forever)
httpxtimeout = httpx.Timeout(0.333, connect=0.333, write=0.333, read=None)


@dataclass(slots=True)
class AwwdioClient:
    url: str = field(default_factory=lambda: os.getenv("ICLI_AWWDIO_URL", None))

    client: httpx.AsyncClient = field(
        default_factory=lambda: httpx.AsyncClient(timeout=httpxtimeout)
    )

    # Dictionary to track the last error time for each say string
    errorHistory: dict[str, int] = field(default_factory=dict)

    # How often to allow logging errors for the same message (in seconds)
    error_throttle_seconds: int = 30

    async def say(
        self, voice: str = "Alex", say: str = "Hello World", speed: int = 250, **kwargs
    ) -> None:
        try:
            await self.client.get(
                f"{self.url}/say",
                params=dict(voice=voice, say=say, speed=speed, prio=-100, **kwargs),
            )
        except Exception as e:
            # Get current time
            now = time.time()

            # Check if we've logged an error for this message recently
            last_error_time = self.errorHistory.get(say, 0)

            # Only log if enough time has passed since the last error for this message
            if now - last_error_time >= self.error_throttle_seconds:
                logger.warning("Speaking failed ({}) for: {}", str(e).lower(), say)
                # Reset last error time for this message
                self.errorHistory[say] = now

    async def sound(self, sound: str = "blip") -> None:
        try:
            await self.client.get(f"{self.url}/play", params=dict(sound=sound))
        except Exception as e:
            logger.warning("Sound failed to send!")
