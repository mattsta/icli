"""Command: add

Category: Live Market Quotes
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.cmds.utils import (
    expand_symbols,
)
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["add"])
@dataclass
class IOpQuotesAdd(IOp):
    """Add live quotes to display."""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*symbols", convert=lambda x: expand_symbols(x))]

    async def run(self):
        keys = await self.state.addQuotes(self.symbols)

        if keys:
            # also update current client persistent quote snapshot
            # (only if we added new quotes...)
            await self.runoplive("qsnapshot")

        return keys
