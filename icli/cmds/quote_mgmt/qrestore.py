"""Command: qrestore

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


@command(names=["qrestore"])
@dataclass
class IOpQuoteRestore(IOp):
    group: str = field(init=False)
    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("group")]

    async def run(self) -> bool:
        """Returns True if we restored quotes, False if no quotes were restored.

        The return value is used to determine whether we load the "default" quote set on startup too.
        """
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)  # type: ignore
        if not symbols:
            logger.error("No quote group found for name: {}", self.group)
            return False

        repopulate = [f'"{x}"' for x in symbols]
        await self.runoplive(
            "add",
            " ".join(repopulate),
        )

        return True
