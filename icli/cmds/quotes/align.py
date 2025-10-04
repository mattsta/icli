"""Command: align

Category: Live Market Quotes
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import asyncio


@command(names=["align"])
@dataclass
class IOpQuotesAlign(IOp):
    """Add a group of commonly used together quotes and spreads all at once.

    This commands adds:
      - ATM strangle
      - ATM ±pts straddle
      - call spread +pts width
      -  put spread -pts width
    """

    symbol: str = field(init=False)
    points: float = field(init=False)
    width: float = field(init=False)
    tradingClass: str = field(init=False)

    def argmap(self):
        return [
            DArg("symbol", convert=str.upper),
            DArg("points", convert=float, default=10),
            DArg("width", convert=float, default=20),
            DArg(
                "tradingClass",
                default="",
                desc="If you need to disambiguate your contracts, you can add a custom trading class. Default: unused",
            ),
        ]

    @property
    def strikeWidthOffset(self) -> float:
        """Offset against the starting points for width calculations.

        e.g. a straddle is points=0, width=0
             a strangle is points=N, width=K, but our commands take (points from ATM) (width from ATM),
               so we need to provide (points + width) for as:
               points=10 width=20 gives the command ATM+10, ATM+30 for the strikes."""
        return self.width + self.points

    async def run(self):
        logger.info(
            "[{}] Using ATM width: {} and strike width: {}",
            self.symbol,
            self.points,
            self.strikeWidthOffset - self.points,
        )

        tradingClassExtension = f"-{self.tradingClass}" if self.tradingClass else ""

        # strangle
        # e.g: straddle /ES 0
        a = self.runoplive("straddle", f"{self.symbol}{tradingClassExtension} 0")

        # straddle
        # e.g.: straddle /ES 10
        b = self.runoplive(
            "straddle", f"{self.symbol}{tradingClassExtension} {self.points}"
        )

        # put spread
        # e.g.: for points=10, width=20 == straddle /ES v p -10 -30
        c = self.runoplive(
            "straddle",
            f"{self.symbol}{tradingClassExtension} v p -{self.points} -{self.strikeWidthOffset}",
        )

        # call spread
        # e.g.: for points=10, width=20 == straddle /ES v c 10 30
        d = self.runoplive(
            "straddle",
            f"{self.symbol}{tradingClassExtension} v c {self.points} {self.strikeWidthOffset}",
        )

        await asyncio.gather(*[a, b, c, d])

        # also update current client persistent quote snapshot
        await self.runoplive("qsnapshot")
