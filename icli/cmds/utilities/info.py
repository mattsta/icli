"""Command: info

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ib_async import (
    Bag,
    Contract,
    FuturesOption,
    Option,
)
from loguru import logger
from mutil.dispatch import DArg
from mutil.frame import printFrame

from icli.cmds.base import IOp, command
from icli.cmds.utils import (
    expand_symbols,
)
from icli.helpers import *

if TYPE_CHECKING:
    pass
import pandas as pd
import prettyprinter as pp  # type: ignore


@command(names=["info"])
@dataclass
class IOpInfo(IOp):
    """Show the underlying IBKR contract object for a symbol.

    This is mainly useful to verify the IBKR details or extract underlying contract IDs
    for other debugging or one-off usage purposes."""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*symbols", convert=lambda x: sorted(expand_symbols(x)))]

    async def run(self):
        contracts = []

        for sym in self.symbols:
            _name, contract = await self.state.positionalQuoteRepopulate(sym)
            contracts.append(contract)

        # remove anything not actually available (typos mainly like trying to look up :!3 which is None)
        contracts = list(filter(None, contracts))

        # Print contracts first because not all contracts qualify for the extra metadata population,
        # but we still want to print something the user requested at least.
        logger.info("Inbound Contracts:\n{}", pp.pformat(contracts))

        # If any lookups fail above, remove 'None' results before we fetch full contracts.
        qcontracts = await self.state.qualify(*contracts)
        logger.info("Qualified Contracts:\n{}", pp.pformat(qcontracts))

        for contract in qcontracts:
            # if contract didn't qualify, we can't do anything more than just print it as we did above
            # (though, bags *never* have a top-level contract id, so we let them pass through)
            if not (isinstance(contract, Bag) or contract.conId):
                continue

            digits = self.state.decimals(contract)

            # Only print ticker if we have an active market data feed already subscribed on this client
            symkey = lookupKey(contract)
            if iticker := self.state.quoteState.get(symkey, None):
                ticker = iticker.ticker

                # The 'pprint' module doesn't use our nice __repr__ override which removes all nan/None fileds (sometimes dozens per ticker),
                # so let's hack around it by printing the formatted dataclass, splitting by comma lines,
                # removing rows with nan values, then just re-assembling it.

                # drop created so it doesn't print
                ticker.created = None  # type: ignore

                assert ticker.contract

                prettyTicker = ",".join(
                    filter(lambda x: "=None" not in x, pp.pformat(ticker).split(","))
                )
                logger.info("Ticker:\n{}", prettyTicker)

                if ticker.histVolatility:
                    logger.info(
                        "[{}] Historical Volatility: {:,.{}f}",
                        ticker.contract.localSymbol,
                        ticker.histVolatility,
                        digits,
                    )

                if ticker.impliedVolatility:
                    logger.info(
                        "[{}] Implied Volatility: {:,.{}f}",
                        ticker.contract.localSymbol,
                        ticker.impliedVolatility,
                        digits,
                    )

                if ticker.histVolatility and ticker.impliedVolatility:
                    if ticker.histVolatility < ticker.impliedVolatility:
                        logger.info(
                            "[{}] Volatility: RISING ({:,.{}f} %)",
                            ticker.contract.localSymbol,
                            100
                            * ((ticker.impliedVolatility / ticker.histVolatility) - 1),
                            digits,
                        )
                    else:
                        logger.info(
                            "[{}] Volatility: FALLING ({:,.{}f} %)",
                            ticker.contract.localSymbol,
                            100
                            * ((ticker.histVolatility / ticker.impliedVolatility) - 1),
                            digits,
                        )

                if ticker.last:
                    logger.info(
                        "[{}] Last: ${:,.{}f} x {}",
                        ticker.contract.localSymbol,
                        ticker.last,
                        digits,
                        int(ticker.lastSize)
                        if int(ticker.lastSize) == ticker.lastSize
                        else ticker.lastSize,
                    )

                # if this is a bag of things, print each underlying symbol too...
                if isinstance(ticker.contract, Bag):
                    logger.info("Bag has {} legs:", len(ticker.contract.comboLegs))
                    legs = zip(
                        ticker.contract.comboLegs,
                        await self.state.qualify(
                            *[
                                Contract(conId=x.conId)
                                for x in ticker.contract.comboLegs
                            ]
                        ),
                    )
                    for legSrc, legContract in legs:
                        logger.info(
                            "    {:>4} {:>3} {:<} ({:<})",
                            legSrc.action,
                            legSrc.ratio,
                            legContract.localSymbol,
                            legContract.localSymbol.replace(" ", ""),
                        )

                    # also provide an easy quote add syntax for moving this around if we want to
                    logger.info(
                        "{}",
                        " ".join(
                            [
                                f"{leg.action} {leg.ratio} {leg.conId}"
                                for leg in ticker.contract.comboLegs
                            ]
                        ),
                    )

                def tickTickBoom(current, prev, name, xchanges=None, xsize=None):
                    # don't print anything if our data is invalid.
                    # invalid can be: NaNs, after hours prices of -1 or 0, etc.
                    if not (current and prev):
                        return

                    udl = "FLAT"
                    amt = current - prev
                    if amt > 0:
                        udl = "UP"
                    elif amt < 0:
                        udl = "DOWN"

                    xchangeDetails = ""

                    if xchanges:
                        sz = int(xsize) if int(xsize) == xsize else xsize
                        xchangeDetails = f" @ ${current:,.{digits}f} x {sz:,} on {len(xchanges)} exchanges"

                    assert ticker.contract
                    logger.info(
                        "[{}] {} tick {} (${:,.{}f}){}",
                        ticker.contract.localSymbol,
                        name,
                        udl,
                        amt,
                        digits,
                        xchangeDetails,
                    )

                tickTickBoom(
                    ticker.bid,
                    ticker.prevBid,
                    "BID",
                    ticker.bidExchange,
                    ticker.bidSize,
                )
                tickTickBoom(
                    ticker.ask,
                    ticker.prevAsk,
                    "ASK",
                    ticker.askExchange,
                    ticker.askSize,
                )
                tickTickBoom(ticker.last, ticker.prevLast, "LAST")

                # protect against ask being -1 or NaN thanks to weird IBKR data issues when markets aren't live
                if ticker.bid and ticker.ask:
                    logger.info(
                        "[{}] Spread: ${:,.{}f} (± ${:,.{}f})",
                        ticker.contract.localSymbol,
                        ticker.ask - ticker.bid,
                        digits,
                        (ticker.ask - ticker.bid) / 2,
                        digits,
                    )

                if ticker.halted:
                    logger.warning("[{}] IS HALTED!", ticker.contract.localSymbol)

                trf = iticker.emaTradeRate.logScoreFrame(0)
                printFrame(
                    trf,
                    f"Trade Rate Stats [scores [prev {iticker.emaTradeRate.diffPrevLogScore:.10f}] [vwap {iticker.emaTradeRate.diffVWAPLogScore:.10f}]]",
                )

                tvf = iticker.emaVolumeRate.logScoreFrame(2)
                printFrame(
                    tvf,
                    f"Volume Rate Stats [scores [prev {iticker.emaVolumeRate.diffPrevLogScore:.10f}] [vwap {iticker.emaVolumeRate.diffVWAPLogScore:.10f}]]",
                )

                if isinstance(iticker.contract, (Option, FuturesOption, Bag)):
                    vfi = iticker.emaIV.logScoreFrame(3)
                    printFrame(
                        vfi,
                        f"IV Stats [scores [prev {iticker.emaIV.diffPrevLogScore:.10f}] [vwap {iticker.emaIV.diffVWAPLogScore:.10f}]]",
                    )

                    dfi = iticker.emaDelta.logScoreFrame(3)
                    printFrame(
                        dfi,
                        f"Delta Stats [scores [prev {iticker.emaDelta.diffPrevLogScore:.10f}] [vwap {iticker.emaDelta.diffVWAPLogScore:.10f}]]",
                    )

                fi = iticker.ema.logScoreFrame(digits)
                printFrame(
                    fi,
                    f"Price Stats [scores [prev {iticker.ema.diffPrevLogScore:.22f}] [vwap {iticker.ema.diffVWAPLogScore:.22f}]]",
                )

                # TODO: have .anaylize return a dataframe too i guess
                qfresults = iticker.quoteflow.analyze()

                logger.info(
                    "[{}] QuoteFlow Duration: {:,.3f} s",
                    ticker.contract.localSymbol,
                    qfresults["duration"],
                )

                logger.info(
                    "[{}] QuoteFlow UP Average Time: {}",
                    ticker.contract.localSymbol,
                    pp.pformat(
                        {
                            f"${k}": f"{v:,.3f} s ({qfresults['uplen'][k]})"
                            for k, v in qfresults["upspeed"].items()
                        }
                    ),
                )
                logger.info(
                    "[{}] QuoteFlow DOWN Average Time: {}",
                    ticker.contract.localSymbol,
                    pp.pformat(
                        {
                            f"${k}": f"{v:,.3f} s ({qfresults['downlen'][k]})"
                            for k, v in qfresults["downspeed"].items()
                        }
                    ),
                )

                for lookback, atr in iticker.atrs.items():
                    logger.info(
                        "[{}] ATR [{}]: {:,.{}f}",
                        ticker.contract.localSymbol,
                        lookback,
                        atr.current,
                        digits,
                    )

                try:
                    # 'statistics' throws an error if there's not enough history yet
                    qs = statistics.quantiles(iticker.history, n=7, method="inclusive")

                    bpos = bisect.bisect_left(qs, iticker.ema[0])
                    qss = [f"{x:,.{digits}f}" for x in qs]
                    qss.insert(bpos, "[X]")

                    low = min(iticker.history)
                    high = max(iticker.history)

                    logger.info(
                        "[{}] stats (from {}): [range {:,.{}f}] [min {:,.{}f}] [max {:,.{}f}] [std {:,.{}f}]",
                        ticker.contract.localSymbol,
                        len(iticker.history),
                        high - low,
                        digits,
                        low,
                        digits,
                        high,
                        digits,
                        statistics.stdev(iticker.history),
                        digits,
                    )

                    logger.info(
                        "[{}] range: {}",
                        ticker.contract.localSymbol,
                        " :: ".join(qss),
                    )
                except:
                    # logger.exception("what?")
                    pass

                if iticker.bags:
                    logger.info(
                        "[{}] Bags ({}): {}",
                        ticker.contract.localSymbol,
                        len(iticker.bags),
                        [x.contract.comboLegs for x in iticker.bags],
                    )

                if iticker.legs:
                    logger.info(
                        "[{}] Tracking Legs ({}):",
                        ticker.contract.localSymbol,
                        len(iticker.legs),
                    )

                    for ratio, leg in iticker.legs:
                        if leg:
                            logger.info("    [ratio {:>2}] {}", ratio, leg.contract)
                        else:
                            logger.warning("    LEG NOT PRESENT")

                # these are filtered to remove None models when they are being populated during startup...
                greeks = {
                    k: v
                    for k, v in {
                        "bid": ticker.bidGreeks,
                        "ask": ticker.askGreeks,
                        "last": ticker.lastGreeks,
                        "model": ticker.modelGreeks,
                    }.items()
                    if v
                }

                df = pd.DataFrame.from_dict(greeks, orient="index")

                # only print our fancy table if it actually exists and if this holds potentially time or volatility risk premium
                if not df.empty and isinstance(
                    ticker.contract, (Bag, Option, FuturesOption)
                ):
                    # remove rows with broken theta values showing up like '-0.000000'
                    # actually, don't do this because we have postive theta on reported spreads and this just drops them all.
                    # df = df[df.theta < -0.0001]
                    # if df.empty:
                    #     continue

                    # make a column for what percentage of theta is the current option price
                    # (basically: your daily rollover loss percentage if the price doesn't move overnight)
                    df["theta%"] = round(df.theta / df.optPrice, 2)
                    df["delta%"] = round(df.delta / df.optPrice, 2)

                    # theta/delta tells you how much the underlying must go up the next day to compensate
                    # for the theta decay.
                    # e.g. delta 0.10 and theta -0.05 means the underlying must go up $0.50 the next day to remain flat.
                    #      delta 0.03 and theta -0.14 means the underlying must go up $5 the next day to remain flat.
                    # technically we should be using charm here (delta for the 'next day'), but this is close enough for now.
                    df["Θ/Δ"] = round(-df.theta / df.delta, 2)

                    # provide rough (ROUGH) estimates for 3 days into the future accounting for theta
                    # (note: theta is negative, so we just add it. also note: theta doesn't decay linearly,
                    #        so these calculations are not _exact_, but it serves as a mental checkpoint to compare against).
                    df["day+1"] = round(df.optPrice + df.theta, 2).clip(lower=0)
                    df["day+2"] = round(df.optPrice + df.theta * 2, 2).clip(lower=0)
                    df["day+3"] = round(df.optPrice + df.theta * 3, 2).clip(lower=0)

                    # remove always empty columns
                    del df["pvDividend"]
                    del df["tickAttrib"]

                    printFrame(df, "Greeks Table")

                    # only show summary greeks if we have more than one greeks row to compare against
                    if len(df) > 1:
                        min_max_mean = df.agg(["min", "max", "mean"])
                        printFrame(min_max_mean, "Summary Greeks Table")
