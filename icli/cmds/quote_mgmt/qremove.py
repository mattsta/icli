"""Command: qremove, qrm

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
import fnmatch


@command(names=["qremove", "qrm"])
@dataclass
class IOpQuoteRemove(IOp):
    """Remove symbols from a quote group."""

    group: str = field(init=False)
    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        nocache = False
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)  # type: ignore
        if not symbols:
            nocache = True
            symbols = self.state.quoteState
            logger.error(
                "[{}] No quote group found so using live quote list...", self.group
            )

        goodbye = set()
        for s in self.symbols:
            for symbol, ticker in symbols.items():
                # guard against running fnmatch on the tuple entries we use for spread quotes
                # (spread quotes can only be removed by :N id)
                if isinstance(symbol, str):
                    if fnmatch.fnmatch(symbol, s):
                        logger.info("Dropping quote: {}", symbol)
                        goodbye.add((symbol, ticker.contract.conId))

        # don't *CREATE* the cache key if we didn't use the cache anyway
        if not nocache:
            symbols -= {x[0] for x in goodbye}  # type: ignore
            self.cache.set(cacheKey, {x[1] for x in goodbye})  # type: ignore

        if not goodbye:
            logger.warning("No matching symbols found?")
            return

        goodbyeIds = [f"{conId}" for _symbol, conId in goodbye]
        logger.info(
            "Removing quotes: {}",
            ", ".join([f"{symbol} -> {conId}" for symbol, conId in goodbye]),
        )

        logger.info("rm {}", " ".join(goodbyeIds))

        await self.runoplive("remove", " ".join(goodbyeIds))
