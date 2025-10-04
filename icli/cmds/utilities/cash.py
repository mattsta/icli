"""Command: cash

Category: Utilities
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mutil.frame import printFrame

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import pandas as pd


@command(names=["cash"])
@dataclass
class IOpCash(IOp):
    def argmap(self):
        return []

    async def run(self):
        result = {
            "Avail Full": [
                self.state.accountStatus["AvailableFunds"],
                self.state.accountStatus["AvailableFunds"] * 2,
                self.state.accountStatus["AvailableFunds"] * 4,
            ],
            "Avail Buffer": [
                self.state.accountStatus["AvailableFunds"] / 1.10,
                self.state.accountStatus["AvailableFunds"] * 2 / 1.10,
                self.state.accountStatus["AvailableFunds"] * 4 / 1.10,
            ],
            "Net Full": [
                self.state.accountStatus["NetLiquidation"],
                self.state.accountStatus["NetLiquidation"] * 2,
                self.state.accountStatus["NetLiquidation"] * 4,
            ],
            "Net Buffer": [
                self.state.accountStatus["NetLiquidation"] / 1.10,
                self.state.accountStatus["NetLiquidation"] * 2 / 1.10,
                self.state.accountStatus["NetLiquidation"] * 4 / 1.10,
            ],
        }

        printFrame(
            pd.DataFrame.from_dict(
                {k: [f"${v:,.2f}" for v in vv] for k, vv in result.items()},
                orient="index",
                columns=["Cash", "Overnight", "Day"],
            )
        )
