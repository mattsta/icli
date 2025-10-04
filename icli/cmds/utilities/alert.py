"""Command: alert

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import prettyprinter as pp  # type: ignore


@command(names=["alert"])
@dataclass
class IOpAlert(IOp):
    """Configure in-icli alert settings for crossovers and level breaches."""

    symbol: str = field(init=False)
    builder: str = field(init=False)
    data: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg("symbol", convert=str.upper),
            DArg(
                "setup",
                desc="How to configure 'data' field for symbol",
                convert=str.lower,
            ),
            DArg(
                "*data",
                desc="Which field of data to edit alert settings for",
                convert=lambda x: list(map(str.lower, x)),
            ),
        ]

    async def run(self):
        foundSymbol, contract = await self.state.positionalQuoteRepopulate(self.symbol)

        if not foundSymbol:
            logger.error("Symbol not found? Can't update alert state!")
            return

        self.symbol = foundSymbol
        assert contract

        logger.info("[{}] Updating alert settings...", self.symbol)

        disable = self.builder in {"off", "no", "false", "0", "disable"}
        enable = not disable

        show = self.builder == "show"

        # one one big OFF message for ALL symbols
        if (not enable) and not self.data:
            ...

        symkey = lookupKey(contract)
        iticker = self.state.quoteState[symkey]

        # else, we have at least one data field to consider
        match self.data[0]:
            case "bar":
                # minor hack to just print the current full alert level state
                if show:
                    logger.info("{}", pp.pformat(iticker.levels))
                    return

                # if sub-fields, set them as populated
                # (this references bar size in seconds, so 86400 == 1 day bar, etc)
                if len(self.data) > 1:
                    for level in self.data[1:]:
                        if found := iticker.levels.get(int(level)):
                            found.enabled = enable
                            logger.info("Now enabled={} for {}", enable, found)
                else:
                    for l in iticker.levels.values():
                        l.enabled = enable
                        logger.info("Now enabled={} for {}", enable, l)
