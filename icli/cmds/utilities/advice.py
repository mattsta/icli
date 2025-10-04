"""Command: advice

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ib_async import (
    Option,
)
from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import numpy as np
import prettyprinter as pp  # type: ignore


@command(names=["advice"])
@dataclass
class IOpAdviceMode(IOp):
    """Generate a market strength score based on active data metrics."""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*symbols",
                desc="Run stats for specific symbol(s) plus default futures",
            )
        ]

    async def run(self):
        # Steps:
        #   - Check current ATM SPX strikes against their VWAP distance
        #   - VIX near HOD or LOD or neutral
        #   - Check EMA crossovers of RTY, ES, NQ for near term and long term directions
        #   - signal for "long the butt" and "short the crown"
        #   - above or below 5 minute volstop
        #   -

        # ATM SPX evaluation
        # 'straddle' returns the contracts for the added straddle/strangle bag and each leg.
        addedContracts = await self.runoplive("straddle", "SPX")

        # Remove the 'Bag' so we only have Options remaining
        spxContracts = filter(lambda x: isinstance(x, Option), addedContracts)

        # now look up tickers for each of the options...
        tickers = [self.state.quoteState[lookupKey(x)] for x in spxContracts]

        callVWAPDistance = 0
        putVWAPDistance = 0

        try:
            for t in tickers:
                match t.contract.right:
                    case "C":
                        callVWAPDistance = (t.bid - t.vwap) / t.vwap
                    case "P":
                        putVWAPDistance = (t.bid - t.vwap) / t.vwap

            logger.info("SPX Call VWAP: {:,.2f} %", callVWAPDistance * 100)
            logger.info("SPX Put VWAP: {:,.2f} %", putVWAPDistance * 100)
        except:
            logger.warning("SPX Strikes VWAP not populated yet... try again soon.")

        # VIX evaluation
        v = self.state.quoteState["VIX"]
        vixHOD = v.last - v.high
        vixLOD = v.last - v.low
        vixVWAPDistance = (v.last - v.vwap) / v.vwap

        logger.info("VIX vwap: {:,.2f} %", vixVWAPDistance * 100)

        # Index crossover checks (EMA and VWAP)
        # fetch contracts for each index
        rty, es, nq = await self.state.qualify(
            *[contractForName(x) for x in "/RTY /ES /NQ".split()]
        )

        # convert contracts to ticker lookup keys
        rtyk, esk, nqk = (lookupKey(x) for x in [rty, es, nq])

        # fetch tickers from lookup keys
        rtyt, est, nqt = (self.state.quoteState[x] for x in [rtyk, esk, nqk])

        def emaCheck(ticker, fast, slow) -> float:
            """Run fast/slow crossover distance for ticker and return the percentage difference."""
            return round(
                100 * (ticker.ema[fast] - ticker.ema[slow]) / ticker.ema[slow], 2
            )

        def tickerScores(ticker: ITicker):
            """Return our custom signals for a single ticker."""
            medium = emaCheck(ticker, 120, 300)
            long = emaCheck(ticker, 300, 1800)

            # current could potentially be None, so check if it exists
            current = ticker.current
            tvw = ticker.vwap

            vwap: float | None = None
            if current:
                vwap = (current - tvw) / tvw

            # get durations for EMA inside ticker, but remove the first and last entries for more stable update checks
            # (first position is just "last updated price" with no trend, and last position is approximate VWAP and doesn't change fast enough)
            durations = ticker.ema.durations[1:-1]

            # fetch emas for each duration
            emas = np.array([ticker.ema[x] for x in durations])

            # "long the butt" detector
            # idea: collect every EMA except the VWAP EMA, if mean(vals) and median(vals) are both less than the shortest and longest ema, we're in a buyable dip.
            emamedian = np.median(emas)
            emamean = np.mean(emas)
            base = (emamedian + emamean) / 2
            e0 = ticker.ema[durations[0]]
            e1 = ticker.ema[durations[-1]]
            longbutt = bool(base <= e0 and base <= e1)
            longscore = round(float(((e0 - base) + (e1 - base)) / 2), 2)

            # "short the crown" detector
            # idea: opposite of "long the butt" where we want the middle values to be higher than the more extreme values.
            shortcrown = bool(base >= e0 and base >= e1)
            shortscore = round(float(((base - e0) + (base - e1)) / 2), 2)

            # Note: for now the quoteflow reporter outputs its own prints
            qfresults = ticker.quoteflow.analyze()

            return dict(
                zip(
                    "vwap medium long butt crown buttscore crownscore quoteflow".split(),
                    [
                        vwap,
                        medium,
                        long,
                        longbutt,
                        shortcrown,
                        longscore,
                        shortscore,
                        qfresults,
                    ],
                )
            )

        # score RTY
        rtystats = tickerScores(rtyt)

        # score ES
        esstats = tickerScores(est)

        # score NQ
        nqstats = tickerScores(nqt)

        stats = dict(
            zip(
                [getattr(x, "localSymbol") for x in [rty, es, nq]],
                [rtystats, esstats, nqstats],
            )
        )

        # 5 minute volstop direction detector
        # bulk-import history for 2 days and run volstop algo for current side, duration, and distance.
        logger.info("Result: {}", pp.pformat(stats))
