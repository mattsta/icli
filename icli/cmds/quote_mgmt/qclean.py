"""Command: qclean

Category: Quote Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import whenever


@command(names=["qclean"])
@dataclass
class IOpQuoteClean(IOp):
    group: str = field(init=False)

    def argmap(self):
        return [DArg("group")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)  # type: ignore
        if not symbols:
            logger.error("No quote group found for name: {}", self.group)
            return

        # Find any expired option symbols and remove them
        remove = []
        now = whenever.ZonedDateTime.now("US/Eastern")

        # if after market close, use today; else use previous day since market is still open
        if (now.hour, now.minute) < (16, 15):
            now = now.subtract(whenever.days(1), disambiguate="compatible")

        datecompare = f"{now.year - 2000}{now.month:02}{now.day:02}"
        for x in symbols:
            if len(x) > 10:
                date = x[-15 : -15 + 6]
                if date <= datecompare:
                    logger.info("Removing expired quote: {}", x)
                    remove.append(f'"{x}"')

        # TODO: fix bug where it's not translating SPX -> SPXW properly for the live removal
        await self.runoplive(
            "qremove",
            "global " + " ".join(remove),
        )
