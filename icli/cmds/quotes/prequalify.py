"""Command: prequalify

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
import dateutil.parser
import whenever


@command(names=["prequalify"])
@dataclass
class IOpPreQualify(IOp):
    """IBKR refuses to qualify valid FOP contracts on the day they expire, but we can cache them ahead of time."""

    symbol: str = field(init=False)
    days: int = field(init=False)
    overwrite: bool = field(init=False)
    verbose: bool = field(init=False)
    tradingClass: str = field(init=False)

    def argmap(self):
        return [
            DArg("symbol", convert=str.upper),
            DArg(
                "days",
                convert=int,
                default=15,
            ),
            DArg("verbose", default=False, convert=lambda x: bool(x)),
            DArg("overwrite", default=False, convert=lambda x: bool(x)),
            DArg(
                "tradingClass",
                default="",
                desc="If you need to disambiguate your contracts, you can add a custom trading class. Default: unused",
            ),
        ]

    async def run(self):
        # Steps:
        #  1. Collect option chain for symbol
        #  2. For each date in chain, and for each strike in each chain, qualify the contract into our cache.

        now = whenever.ZonedDateTime.now("US/Eastern")
        nowdate = now.date()

        highestdate = now.add(
            whenever.days(self.days), disambiguate="compatible"
        ).date()

        symbol = self.symbol
        chains = await self.runoplive("chain", symbol)

        for sym, chain in chains.items():
            logger.info(
                "[{}] Qualifying contracts for dates between {} and {}: {}",
                sym,
                nowdate,
                highestdate,
                chain.keys(),
            )

            for date, strikes in chain.items():
                pdate = whenever.LocalDateTime.from_py_datetime(
                    dateutil.parser.parse(date)
                ).date()

                # don't qualify the past...
                if pdate < nowdate:
                    continue

                # don't qualify expired contracts if our cached chains have old dates
                if pdate > highestdate:
                    continue

                # date format is YYMMDD so remove the first '20'
                datefmt = date[2:]
                names = []
                tradingClassExtension = (
                    f"-{self.tradingClass}" if self.tradingClass else ""
                )
                for strike in strikes:
                    # Note: we use the INPUT symbol because it will have the proper type designator.
                    # (e.g. we need /ES... and not ES... and 'sym' from the cache is the underlying symbol. We should probably fix that too to include the contract type)
                    names.extend(
                        [
                            f"{symbol}{datefmt}P{int(strike * 1000):08}{tradingClassExtension}",
                            f"{symbol}{datefmt}C{int(strike * 1000):08}{tradingClassExtension}",
                        ]
                    )

                # run qualification batch at each date instead of just runnin them ALL at the end
                logger.info(
                    "[{} :: {}] Qualifying {} contracts (overwrite: {})...",
                    symbol,
                    date,
                    len(names),
                    self.overwrite,
                )

                got = await self.state.qualify(
                    *[contractForName(name) for name in names], overwrite=self.overwrite
                )

                if self.verbose:
                    logger.info("{}", got)
