"""Command: qsave

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


@command(names=["qsave"])
@dataclass
class IOpQuoteSave(IOp):
    group: str = field(init=False)
    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        self.cache.set(cacheKey, set(self.symbols))  # type: ignore
        logger.info("[{}] {}", self.group, self.symbols)

        repopulate = [f'"{x}"' for x in self.symbols]
        await self.runoplive(
            "add",
            " ".join(repopulate),
        )
