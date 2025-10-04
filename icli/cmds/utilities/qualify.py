"""Command: qualify

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.cmds.utils import (
    expand_symbols,
)
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["qualify"])
@dataclass
class IOpQualify(IOp):
    """Qualify contracts in the cache for future usage.

    Current reason this exists: IBKR apparently refuses to qualify 0dte /ES contracts, so we
    look them *all* up the day before so we can access them live the next day."""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*symbols", convert=lambda x: sorted(expand_symbols(x)))]

    async def run(self):
        # build contract objects from names
        logger.info("Qualifying for names: {}", self.symbols)

        contracts = [contractForName(s) for s in self.symbols]
        logger.info("Qualifying: {}", contracts)

        # ask our cache or IBKR for the complete details including unique contract ids
        got = await self.state.qualify(*contracts)

        # results
        logger.info("Result: {}", got)
        return got
