"""Command: reporter

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.bgtask import BGSchedule
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import sys


@command(names=["reporter"])
@dataclass
class IOpMarketReporter(IOp):
    """Generate a periodic market strength report using internal data (emas, trade/volume, volatility) for positioning recommendations."""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*symbols",
                desc="Enable periodic directional state reporting for specific symbols",
            )
        ]

    async def run(self):
        # lookup vix iticker for sharing volatility reporting everywhere
        vf, vc = await self.state.positionalQuoteRepopulate("I:VIX")
        assert vc
        vkey = lookupKey(vc)
        assert vkey
        vix = self.state.quoteState.get(vkey)
        assert vix

        async def report(iticker):
            """Report on symbol when called."""

            # report on sweeping EMA combinations (pair-wise directional directions, if all agree, major direction advantage, also scale to "age" of ticker so we don't report durations longer than the ticker has lived)
            # report on suggested instruments to use given strength
            # perhaps include dependent quotes for underlying symbol? If asking for ES direction, check our ES options for their directional strength too?
            # Difference between running this on underlying vs options for side strength?

            # Position Strength Recommendations:
            #  - DELTA-REDUCED FUTURES POSITION MULTIPLES
            #  - VOLATILITY ATM STRADDLES ON RISING VIX
            #  - DIRECTIONAL-WEIGHTED VOLATILITY STRADDLES ON LOWER VIX, RISING UNDERLYING

            age = iticker.age
            sym = iticker.contract.localSymbol

            def positionSuggestion(scores):
                vixReport = vix.rms()

                suggest = "NONE YET"

                # if vix is DECLINING, we can go directional long.
                vr300 = vixReport[300]
                if vr300 <= 0:
                    suggest = "VIX DOWN, SUGGEST DIRECTIONAL LONG"
                elif vr300 >= 0:
                    suggest = "VIX UP, SUGGEST VOLATILITY STRADDLES"

                pass

        for symbol in self.symbols:
            # Step 1: Run First Report
            # Step 2: Create reporter for symbol
            # Step 3: Schedule Continuous Reporter

            found, contract = await self.state.positionalQuoteRepopulate(symbol)

            if not found:
                logger.error("[{}] Not found?", symbol)
                continue

            assert contract

            symkey = lookupKey(contract)
            iticker = self.state.quoteState.get(symkey)

            created = self.task_create(
                f"market direction reporter for {symbol}",
                report(iticker),
                schedule=BGSchedule(
                    delay=0, runtimes=int(sys.float_info.max), pause=90
                ),
            )

            logger.info("[{}] Created recurring reporting task: {}", symbol, created)
