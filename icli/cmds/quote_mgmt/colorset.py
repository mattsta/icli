"""Command: colorset

Category: Quote Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["colorset"])
@dataclass
class IOpColors(IOp):
    """Select a new color scheme either by collection name or with direct colors."""

    style: str = field(init=False)

    def argmap(self):
        return [DArg("style")]

    async def run(self):
        cacheKey = ("colors", f"client-{self.state.clientId}")
        cacheVal = dict(toolbar=self.style)
        self.cache.set(cacheKey, {"colors": cacheVal})  # type: ignore

        self.state.updateToolbarStyle(self.style)
