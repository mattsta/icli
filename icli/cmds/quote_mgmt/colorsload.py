"""Command: colorsload

Category: Quote Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["colorsload"])
@dataclass
class IOpColorsLoad(IOp):
    otherClientId: int = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*otherClientId",
                convert=lambda x: int(x) if x else self.state.clientId,
                desc="Optional Client ID if loading another client's color setting",
            )
        ]

    async def run(self):
        cacheKey = ("colors", f"client-{self.otherClientId}")
        cons = self.cache.get(cacheKey)  # type: ignore
        if not cons:
            return False

        try:
            # snapshots are always saved with exact Contract objects, so we can just restore them directly
            cs = cons.get("colors", {}).get("toolbar")
            self.state.updateToolbarStyle(cs)
            return True
        except:
            pass

        return False
