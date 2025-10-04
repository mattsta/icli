"""A refactor-base for splitting out common helpers between cli and lang"""

from __future__ import annotations

import asyncio
import bisect
import enum
import functools
import locale
import math
import platform
import re
import statistics
import time
import types

import dateutil
import numpy as np
import pandas as pd

ourjson: types.ModuleType
# Only use orjson under CPython, else use default json (because `json` under pypy is faster than orjson)
# (also we are doing this "import name, assign name to global" to get around mypy complaining about double importing with the same alias)
if platform.python_implementation() == "CPython":
    import orjson

    ourjson = orjson
else:
    import json

    ourjson = json

import datetime
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from functools import cached_property
from typing import *

import httpx
import ib_async  # just for UNSET_DOUBLE
import questionary
import tradeapis.cal as tcal
import websockets
from cachetools import cached
from dotenv import dotenv_values
from ib_async import (
    CFD,
    Bag,
    Bond,
    Commodity,
    ContFuture,
    Contract,
    Crypto,
    Forex,
    Future,
    FuturesOption,
    Index,
    MutualFund,
    Option,
    OptionComputation,
    Order,
    Stock,
    Ticker,
    Trade,
    Warrant,
)
from loguru import logger
from questionary import Choice
from tradeapis.orderlang import (
    DecimalCash,
    DecimalPercent,
    DecimalShares,
    OrderIntent,
)

from icli.futsexchanges import FUTS_EXCHANGE

from .tinyalgo import ATRLive

# auto-detect next index futures expiration month based on roll date
# we add some padding to the futs exp to compensate for having the client open a couple days before
# (which will be weekends or sunday night, which is fine)
futexp: Final = tcal.nextFuturesRollDate(
    datetime.datetime.now().date() + datetime.timedelta(days=2)
)

nan: Final = float("nan")

# Also map for user typing shorthand on command line order entry.
# Values abbreviations are allowed for easier command typing support.
# NOTE: THIS LIST IS USED TO TRIGGER ORDERS IN order.py:IOrder().order() so THESE NAMES MUST MATCH
#       THE OFFICIAL IBKR ALGO NAME MAPPINGS THERE.
# This is a TRANSLATION TABLE between our "nice" names like 'AF' and the IBKR ALGO NAMES USED FOR ORDER PLACING.
# NOTE: DO NOT SEND IOrder.order() requests using 'AF' because it must be LOOKED UP HERE FIRST.
# TODO: on startup, we should assert each of these algo names match an actual implemented algo order method in IOrder().order()
ALGOMAP: Final = dict(
    LMT="LMT",
    LIM="LMT",
    LIMIT="LMT",
    AF="LMT + ADAPTIVE + FAST",
    AS="LMT + ADAPTIVE + SLOW",
    MID="MIDPRICE",
    MIDPRICE="MIDPRICE",
    SNAPMID="SNAP MID",
    SNAPMKT="SNAP MKT",
    SNAPREL="SNAP PRIM",
    SNAPPRIM="SNAP PRIM",
    MTL="MTL",  # MARKET-TO-LIMIT (execute at top-of-book, but don't sweep, just set a limit for remainder)
    PRTMKT="MKT PRT",  # MARKET-PROTECT (futs only), triggers immediately
    PRTSTOP="STP PRT",  # STOP WITH PROTECTION (futs only), triggers when price hits
    PRTSTP="STP PRT",  # STOP WITH PROTECTION (futs only), triggers when price hits
    PEGMID="PEG MID",  # Floating midpoint peg, must be directed IBKRATS or IBUSOPT
    REL="REL",
    STOP="STP",
    STP="STP",
    STPLMT="STP LMT",
    STP_LMT="STP LMT",
    TSL="TRAIL LIMIT",
    MKT="MKT",
    MIT="MIT",
    LIT="LIT",
    AFM="MKT + ADAPTIVE + FAST",
    AMF="MKT + ADAPTIVE + FAST",
    ASM="MKT + ADAPTIVE + SLOW",
    AMS="MKT + ADAPTIVE + SLOW",
    MOO="MOO",
    MOC="MOC",
)

D100: Final = Decimal("100")
DN1: Final = Decimal("-1")
DP1: Final = Decimal("1")

# Also compare: https://www.cmegroup.com/trading/equity-index/rolldates.html
logger.info("Futures Next Roll-Forward Date: {}", futexp)
FU_DEFAULT = dict(ICLI_FUT_EXP=f"{futexp.year}{futexp.month:02}")  # YM like: 202309
FU_CONFIG = {**FU_DEFAULT, **dotenv_values(".env.icli"), **os.environ}  # type: ignore

# TEMPORARY OVERRIDE FOR EXPIRATION WEEK OPTION PROBLEMS
# TODO: for futures options, we need to reead the detail "uynderlying contract" to use for distance and quotes instead of the live symbol. sigh.
FUT_EXP = FU_DEFAULT["ICLI_FUT_EXP"]
# FUT_EXP = "202409"

FUTS_MONTH_MAPPING: Final = {
    "F": "01",  # January
    "G": "02",  # February
    "H": "03",  # March
    "J": "04",  # April
    "K": "05",  # May
    "M": "06",  # June
    "N": "07",  # July
    "Q": "08",  # August
    "U": "09",  # September
    "V": "10",  # October
    "X": "11",  # November
    "Z": "12",  # December
}

PQ: Final = enum.Enum("PQ", "PRICE QTY")

type BuySell = Literal["BUY", "SELL"]

type ContractId = int


def fmtmoney(val: float | int | Decimal):
    """Return a formatted money string _with_ a comma in them for thousands separator."""
    return locale.currency(val, grouping=True)


@dataclass(slots=True)
class FillReport:
    """A commission report of a filled execution.

    We can have multiple executions per 'orderId' through time or even across symbols for bags.
    """

    orderId: int
    conId: int
    sym: str
    side: str  # execution side (buy/sell etc.)
    shares: float  # number of shares traded
    price: float  # fill price
    pnl: float  # realized profit/loss
    commission: float  # commission paid
    when: datetime.datetime  # time the trade was executed

    @property
    def qty(self) -> float:
        """Return +qty for longs and -qty for shorts"""
        if self.side == "BOT":
            return self.shares

        assert self.side == "SLD"
        return -self.shares


type FPrice = float
type MaybePrice = FPrice | None
type PercentAmount = tuple[MaybePrice, MaybePrice]
type Seconds = int

RTH_EMA_VWAP: Final = 23_400


@dataclass(slots=True, frozen=True)
class TradeOrder:
    """Just holde a trade/order combination pair for results reporting."""

    trade: Trade
    order: Order


@dataclass(slots=True, frozen=True)
class FullOrderPlacementRecord:
    limit: TradeOrder
    profit: TradeOrder | None = None
    loss: TradeOrder | None = None


@dataclass(slots=True)
class PaperLog:
    """Simplified paper trading log with P&L tracking."""

    _trades: list[dict[str, float]] = field(default_factory=list)

    def log(self, size: float, price: float):
        """
        Record a new paper trade.

        Args:
            size (float): Trade size (positive for long, negative for short)
            price (float): Execution price
        """
        if (
            not isinstance(size, (int, float))
            or not isinstance(price, (int, float))
            or price <= 0
        ):
            raise ValueError("Invalid trade parameters")

        self._trades.append({"size": size, "price": price})

    def report(self, current_price: float | None = None) -> dict[str, float | None]:
        """
        Generate a comprehensive trading report.

        Args:
            current_price (Optional[float]): Current market price for unrealized P&L

        Returns:
            dict: Detailed trading report
        """
        if not self._trades:
            return {
                "total_size": 0,
                "average_price": None,
                "total_cost": 0,
                "realized_pl": 0,
                "unrealized_pl": None,
                "total_pl": 0,
            }

        # Calculate total position and cost
        total_size = sum(trade["size"] for trade in self._trades)
        total_cost = sum(trade["size"] * trade["price"] for trade in self._trades)

        # Calculate average price
        average_price = total_cost / total_size if total_size != 0 else None

        # Realized P&L (profit from closed trades)
        realized_pl = self._calculate_realized_pl()

        # Unrealized P&L
        unrealized_pl = None
        if current_price is not None and total_size != 0:
            unrealized_pl = (current_price - average_price) * total_size  # type: ignore

        return {
            "total_size": total_size,
            "average_price": average_price,
            "total_cost": round(total_cost, 4) if total_cost else None,
            "realized_pl": round(realized_pl, 4) if realized_pl else None,
            "unrealized_pl": round(unrealized_pl, 4) if unrealized_pl else None,
            "total_pl": round(realized_pl + (unrealized_pl or 0), 4),
        }

    def _calculate_realized_pl(self) -> float:
        """
        Calculate realized profit/loss by matching opposite trades.

        Returns:
            float: Total realized profit/loss
        """
        realized_pl = 0

        for t in self._trades:
            realized_pl += t["size"] * t["price"]  # type: ignore

        return -realized_pl

    def reset(self):
        """Clear all trade history."""
        self._trades.clear()


@dataclass(slots=True)
class TWEMA:
    """Time-Weighted EMA for when we have un-equal event arrival, but we want to still collect events-over-time.

    (e.g. we can't just have an EMA of "last N data points" because datapoints could be arriving in 250ms or 3 s or 15 s or 300s...
    """

    # EMA durations in seconds
    # 3,900 seconds is 65 minutes; 23_400 seconds is 6.5 hours (390 minutes)
    durations: tuple[int, ...] = (
        0,  # we use '0' to mean "last value seen"
        15,
        30,
        60,
        120,
        180,
        300,
        900,
        1800,
        3_900,
        RTH_EMA_VWAP,
    )

    # actual EMA values
    # Dict is format [EMA duration in seconds, EMA value]
    emas: dict[int, float] = field(default_factory=dict)

    # metadata EMAs
    diffVWAP: dict[int, float] = field(default_factory=dict)
    diffVWAPLog: dict[int, float] = field(default_factory=dict)
    diffPrevLog: dict[int, float] = field(default_factory=dict)

    # metadata scores
    diffVWAPLogScore: float = 0.0
    diffPrevLogScore: float = 0.0

    # i put emas in ur emas
    diffVWAPLogScoreEMA: dict[int, float] = field(default_factory=dict)
    diffPrevLogScoreEMA: dict[int, float] = field(default_factory=dict)

    last_update: float = 0

    def __post_init__(self) -> None:
        # Just verify durations are ALWAYS sorted from smallest to largest
        self.durations = tuple(sorted(self.durations))

    def update(self, new_value: float | None, timestamp: float) -> None:
        if new_value is None:
            return

        if self.last_update == 0:
            self.last_update = timestamp

            for duration in self.durations:
                self.emas[duration] = new_value

            return

        time_diff = timestamp - self.last_update
        self.last_update = timestamp

        # update all EMAs
        # Use position 0 to store the current "live" input value without any adjustments.
        self.emas[0] = new_value

        # skip the 0th entry because we manully write into it

        for period in self.durations[1:]:
            value = self.emas[period]
            alpha = 1 - math.exp(-time_diff / period)
            last = alpha * new_value + (1 - alpha) * value
            self.emas[period] = last

        # now update difference EMAs:
        #  - price differences (from previous)
        #  - difference from VWAP (longest duration)
        #  - difference from previous

        # can't update logs of negative prices if we init weird
        # (things like NYSE-TICK have negative "price" ranges)
        # if last <= 0:
        #    return

        # VWAP vs. Current comparisons
        # loglast = math.log(last)
        self.diffVWAPLogScore = 0.0
        for k in self.durations:
            v = self.emas[k]

            # same check against negative prices here too...
            # if v <= 0:
            #     return

            # price difference VWAP
            self.diffVWAP[k] = v - last

            # log difference VWAP
            # dvl = 100 * (math.log(v) - loglast)
            dvl = 100 * ((v - last) / (last or 1))
            self.diffVWAPLog[k] = dvl

            if k > 0:
                self.diffVWAPLogScore += (1 / k) * dvl

        # Previous vs. Current comparisons
        # process differences from high to low, so reverse, because
        # we know the dict keys are _already_ in sorted order from lowest to highest.
        prev = 0.0
        self.diffPrevLogScore = 0.0
        for k in reversed(self.durations):
            # here = math.log(self.emas[k])
            here = self.emas[k]
            if not prev:
                prev = here
                continue

            # log difference vs previous EMA price
            dpl = 100 * ((here - prev) / (prev or 1))
            prev = here

            self.diffPrevLog[k] = dpl

            if k > 0:
                self.diffPrevLogScore += (1 / k) * dpl

        self.updateDiffEMAs(time_diff)

    def updateDiffEMAs(self, time_diff: float):
        prevLog = self.diffPrevLogScore
        vwapLog = self.diffVWAPLogScore

        # if first run, just set both of them then wait for more updates
        if not self.diffPrevLogScoreEMA:
            for duration in self.durations:
                self.diffPrevLogScoreEMA[duration] = prevLog
                self.diffVWAPLogScoreEMA[duration] = vwapLog

            return

        durationsWithoutZero = self.durations[1:]

        # update all EMAs
        for period in durationsWithoutZero:
            value = self.diffPrevLogScoreEMA[period]
            alpha = 1 - math.exp(-time_diff / period)
            self.diffPrevLogScoreEMA[period] = last = (
                alpha * prevLog + (1 - alpha) * value
            )

        for period in durationsWithoutZero:
            value = self.diffVWAPLogScoreEMA[period]
            alpha = 1 - math.exp(-time_diff / period)
            self.diffVWAPLogScoreEMA[period] = last = (
                alpha * vwapLog + (1 - alpha) * value
            )

        # set current values as position zero...
        self.diffPrevLogScoreEMA[0] = prevLog
        self.diffVWAPLogScoreEMA[0] = vwapLog

    def __getitem__(self, idx) -> float:
        return self.emas.get(idx, 0)

    def get(self, idx, default=None) -> float:
        return self.emas.get(idx, default)

    def rms(self) -> dict[int, float]:
        """Calculate RMS for each slice of the EMAs going higher and higher"""

        # verify order is correct for our math to work
        # This is just walking a [(lookback, ema)] list of stuff in depth-adjusted inputs all the way down.
        sema = sorted(self.emas.items(), reverse=True)

        def genscore(threshold):
            """Generate an adaptive end-start RMS score for each step of the EMA lookback"""
            use = list(filter(lambda x: x[0] <= threshold, sema))

            # if only one element matched, we can't compare it against itself, so the result is always zero.
            if len(use) == 1:
                return 0

            _idxs, emas = zip(*use)
            scores = rmsnorm(emas)
            return scores[-1] - scores[0]

        scores = {}
        for k, v in sema:
            scores[k] = genscore(k)

        return scores

    def logScoreFrame(self, digits: int = 2) -> pd.DataFrame:
        rms = self.rms()
        return pd.DataFrame(
            dict(
                prevlog={k: round(v, 4) for k, v in reversed(self.diffPrevLog.items())},
                prevscore={k: v * 1000 for k, v in self.diffPrevLogScoreEMA.items()},
                vwaplog={k: round(v, 4) for k, v in self.diffVWAPLog.items()},
                vwapscore={k: v * 1000 for k, v in self.diffVWAPLogScoreEMA.items()},
                ema={k: round(v, digits) for k, v in self.emas.items()},
                vwapdiff={k: round(v, digits) for k, v in self.diffVWAP.items()},
                rms={k: round(v, 6) for k, v in rms.items()},
            )
        )


@dataclass(slots=True, frozen=True)
class QuoteSizes:
    """A holder for passing around price, bid, ask, and size details."""

    bid: float | None
    ask: float | None
    bidSize: float | None
    askSize: float | None
    last: float | None
    close: float | None

    @property
    def current(self) -> float | None:
        bid = self.bid
        ask = self.ask

        if bid is not None and ask is not None:
            return (bid + ask) / 2

        # Note: don't use '.last or .close' here because on startup IBKR
        #       loads values aync, so sometimes 'close' appears before 'last'
        #       and we don't want to return potentially 1-3 day old 'close' values
        #       when we want the most recentHistoryAnchorly last traded price (even if we have to
        #       wait an extra update tick or two for it to arrive)
        # Though, there is a weird bug with the SPX ticker where, after hours, when you
        # subscribe it gives you two immediate values: the CORRECT value for the close ast 'last'
        # then an oddly incorrect value off by 1-3 points as a new 'last', so if you subscribe to SPX
        # after hours, you get two 'last' price updates and they conflict. No idea why, but even their
        # official app values show the "incorrect SPX" price after close instead of the final price.
        return ask if ask is not None else self.last


@dataclass(slots=True)
class LevelLevels:
    """Store a mapping of type (sma, volume?, etc?) and lookback duration (seconds) to level breaching price (price)."""

    levelType: str
    lookback: int
    lookbackName: str
    level: float

    def __post_init__(self) -> None:
        # use native python floats instead of allowing numpy floats to sneak in here
        self.level = float(self.level)

        # don't allow names to be "open" because our spoken events use 'OPEN' when positions are created
        # (it's confusing to have a _price alert_ say OPEN as well as an _order alert_ also use the same keyword)
        if self.lookbackName == "open":
            self.lookbackName = "start"


@dataclass(slots=True)
class LevelBreacher:
    """Store a collection of levels generated from a bar size with appropriate levels.

    e.g. collection: SMA
         bar size: 1 day
         levels: [<5, price>, <10, price>, <20, price>, <60, price>, <220, price>, <325, price>]
    """

    duration: int
    levels: list[LevelLevels] = field(default_factory=list)

    durationName: str = field(init=False)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.durationName = convert_time(self.duration)


@dataclass(slots=True, frozen=True)
class QuoteFlowPoint:
    bid: float
    ask: float
    timestamp: float


@dataclass(slots=True)
class QuoteFlow:
    """Track the progress of bid/ask levels over time.

    This helps us see how quickly prices are moving.

    The goal is to detect how quickly bids are growing larger than previous asks (or the opposite, when asks are falling below bids).
    """

    # a 1,200 entry history gives us 5 minutes of price history at 250 ms updates
    pairs: deque[QuoteFlowPoint] = field(default_factory=lambda: deque(maxlen=1_200))

    def update(self, bid, ask, timestamp):
        """Save the current bid/ask/timestamp into our history for analyzing price direction."""
        self.pairs.append(QuoteFlowPoint(bid, ask, timestamp))

    def analyze(self):
        """Walk every recorded 'pairs' to figure out how long it takes for either a bid to become the ask or an ask to become a bid."""

        if not self.pairs:
            return defaultdict(float)

        # mapping of price difference to previous point seen for next comparison
        prevpoints: dict[float, QuoteFlowPoint] = {}

        updoot: dict[float, list[float]] = defaultdict(list)
        downdoot: dict[float, list[float]] = defaultdict(list)
        ranges = (0, 0.5, 1, 3, 5, 15)
        for p in self.pairs:
            match p:
                case QuoteFlowPoint(bid=bid, ask=ask, timestamp=timestamp) as qfp:
                    # only attempt to use valid quotes
                    if not (bid and ask):
                        continue

                    # if this is the first attempt, we want to initialize every previous value with the current value
                    if not prevpoints:
                        for r in ranges:
                            prevpoints[r] = qfp

                        continue

                    for r in ranges:
                        prevpoint = prevpoints[r]

                        if bid - prevpoint.ask >= r:
                            # price is RISING because bid is now above PREVIOUS ASK
                            updoot[r].append(timestamp - prevpoint.timestamp)

                            # update previous point since we USED it for date (otherwise the intermediate parts didn't breach)
                            prevpoints[r] = qfp
                        elif prevpoint.bid - ask >= r:
                            # price is FALLING because previous ask is BELOW current BID
                            downdoot[r].append(timestamp - prevpoint.timestamp)
                            prevpoints[r] = qfp

        # return time between last data (most recent) and first data (oldest)
        duration = self.pairs[-1].timestamp - self.pairs[0].timestamp

        upspeeds: dict[float, float] = dict()
        for r, l in updoot.items():
            upspeeds[r] = statistics.mean(l)

        downspeeds: dict[float, float] = dict()
        for r, l in downdoot.items():
            downspeeds[r] = statistics.mean(l)

        # TODO: also include stats about the DISTANCE of the breaches (0.10 cents? $10? we need to be generating better sub-stats per price range)
        # TODO: we ALSO need to populate a metric for "the current trend" if prices are NOW moving up (or the last timestamp a price moved up or down individually)
        # TODO: also try to not double count the same side if conditions remain in-range but not moving?
        # TODO: return this as a dataframe? rows are the price blocks, columns are uplen/upspeed/downlen/downspeed?
        return dict(
            duration=duration,
            uplen={k: len(v) for k, v in updoot.items()},
            upspeed=upspeeds,
            downlen={k: len(v) for k, v in downdoot.items()},
            downspeed=downspeeds,
        )


@dataclass(slots=True)
class ITicker:
    """Our own version of a ticker with more composite and self-reporting details."""

    # underlying live-updating Ticker object
    ticker: Ticker

    # map of cached contract ids to contracts
    # (globally shared contract cache)
    state: Any

    # proper contract name is populated on instance creation
    name: str = "UNDEFINED"

    # just a recentHistoryAnchor price history for every new price update.
    # length assumes we get 4 price updates per second and we want 1 minute of history.
    history: deque[float] = field(default_factory=lambda: deque(maxlen=60 * 4))

    # hold EMA of instrument price over different time periods.
    ema: TWEMA = field(default_factory=TWEMA)

    # For options, also track some extra fields over time...
    emaIV: TWEMA = field(default_factory=TWEMA)
    emaDelta: TWEMA = field(default_factory=TWEMA)
    emaVega: TWEMA = field(default_factory=TWEMA)

    # ticker stats history
    emaTradeRate: TWEMA = field(default_factory=TWEMA)
    emaVolumeRate: TWEMA = field(default_factory=TWEMA)

    # synthetic ATR using our own accounting.
    # calculate live ATR based on quote updates
    atrs: dict[int, ATRLive] = field(default_factory=dict)

    # if this is a spread, we track individual legs.
    # The format is (ratio, ITicker).
    # Short legs will have a negative ratio number so the math works out.
    # Note: these are tuple() because they must all be updated at the same time instead of mutated.
    legs: tuple[tuple[int, ITicker | None], ...] = tuple()

    # if this is a single contract CURRENTLY PARTICIPATING IN OPEN BAG QUOTES, track all bags here too.
    # We want to track the bags so when a leg contract ticks, we update all bag quotes at the same time.
    bags: set[ITicker] = field(default_factory=set)

    # if spread, the "width" of the spread (may not be valid for more than 2 legs, but we try)
    width: float | int = 0

    alerts: dict[int | str, dict[float, tuple[float, float] | None]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(None.__class__))
    )

    alertDelta: float = 0.0
    alertIV: float = 0.0

    # 'levels' map from a bar duration (2 minute, 5 minute, 30 minute, 1 hour, 1 day, 1 week) to the level records holder
    levels: dict[int, LevelBreacher] = field(default_factory=dict)

    # just track an estimate of current direction given short-term EMA crossovers (or maybe log scores too)
    prevDirUp: bool | None = None

    quoteflow: QuoteFlow = field(default_factory=QuoteFlow)

    created: float = field(default_factory=time.time)

    def __post_init__(self):
        assert self.ticker.contract
        self.name = self.state.nameForContract(self.ticker.contract)

        # by default, give all instruments a delta=1 greek designation
        # (if the instrument is a live option, this will be overwritten immediately with live data)
        if not isinstance(self.ticker.contract, (Option, FuturesOption)):
            self.ticker.modelGreeks = OptionComputation(0, 0, 1, 0, 0, 0, 0, 0, 0)

        # init ATRs at our default allowances
        # (the .25 is because fully active quotes update in 250 ms intervals, so we normalize "events per second" by update frequency)
        # (ergo, this is a 90 second vs. 45 second ATR)
        for lookback in (90, 120, 180, 300, 420, 600, 840, 900, 1260, 1800):
            self.atrs[lookback] = ATRLive(
                int(lookback / 0.25), int(lookback / 2 / 0.25)
            )

    def __hash__(self) -> int:
        return hash(self.ticker)

    def __getattr__(self, key):
        """For any non-existing property requested, just fetch from the inner Ticker itself."""
        return getattr(self.ticker, key)

    @property
    def age(self) -> float:
        return time.time() - self.created

    def quote(self) -> QuoteSizes:
        t = self.ticker
        bid = t.bid
        ask = t.ask
        bidSize = t.bidSize
        askSize = t.askSize
        last = t.last
        close = t.close

        # if this is a bag, we can generate some more synthetic details than is provided elsewhere...
        if isinstance(t.contract, Bag):
            if False:
                # We *could* do this, but is the current method of EMA-if-no-VWAP okay enough?
                vwap = 0

                # always set ticker vwap for spread  into ticker object
                for ratio, quote in self.legs:
                    vwap += (quote.vwap or nan) * ratio

                t.vwap = vwap

            # optionally populate bid/ask and size details if current spread isn't getting quotes for some reason
            if bid is None and ask is None:
                bid = 0
                ask = 0
                bidSize = float("inf")
                askSize = float("inf")
                for ratio, quote in self.legs:
                    # if a quote doesn't exist, we need to abandon trying to generate any part of this synthetic quote
                    # because we don't have all the data we need so just combining partial values would be wrong.
                    if not (quote and quote.ask and quote.bid):
                        bid = 0
                        ask = 0
                        vwap = nan
                        bidSize = 0
                        askSize = 0
                        usePrice = None
                        break

                    # SELL legs have opposite signs and positions because they are credits
                    bid += quote.bid * ratio
                    ask += quote.ask * ratio

                    # the "quantity" of a spread is the smallest number available for the combinations
                    bidSize = min(bidSize, quote.askSize)
                    askSize = min(askSize, quote.bidSize)

        # default: return current values
        return QuoteSizes(bid, ask, bidSize, askSize, last, close)

    @property
    def vwap(self) -> float:
        """Override "vwap" paramter option for ITicker allowing us to return our synthetic ema vwap if vwap doesn't exist."""

        if vwap := self.ticker.vwap:
            return vwap

        # return our "daily hours EMA" which should _approximate_ a VWAP for most instruments.
        return self.ema[RTH_EMA_VWAP]

    @property
    def current(self) -> float | None:
        return self.quote().current

    def processTickerUpdate(self) -> None:
        """Update data for this ticker and any dependent tickers when we received new data."""

        q = self.quote()
        current = q.current

        # if no bid/ask/last/close exists, we don't have anything else useful to do here...
        if current is None:
            return

        # log current price update into history...
        # (this also handles updating abandon bag quotes with synthetic quotes from the active legs)
        self.history.append(current)

        for atr in self.atrs.values():
            atr.update(current)

        self.quoteflow.update(self.ticker.bid, self.ticker.ask, self.ticker.timestamp)

        # IBKR spreads only update high/low values when the exact spread is executed, but we can track
        # more detailed high/lows based on current midpoints as they occur (at least until a restart
        # then we begin the process again).
        # Note: the IBKR API can still overwrite these high/low values with active trades when they happen
        #       which may be lower highs or higher lows than the midpoints we've tracked along the way.
        # We could also do this for regular Option and FuturesOption contracts too, but for now we're leaving those as live high/low reports.
        if isinstance(self.ticker.contract, Bag):
            # Don't allow bad negative quotes to pollute the high/low feed counter to the direction of the spread.
            # Logic is: only allow setting positive prices on positive spreads, negative prices on negative spreads, etc
            ema = self.ema[60]

            # also limit these updates to if bid and ask are within 40% of each other to avoid misquoted bouncing on extreme ranges
            # (positive spreads have bid < ask; negative spreads have bid > ask)
            if (
                # current > 0 and ema > 0 and (q.bid and q.ask and q.bid / q.ask >= 0.60)
                current > 0
                and ema > 0
                and q.bid
                and q.ask
                and (abs(q.bid - q.ask) <= 5)
            ) or (
                # current < 0 and ema < 0 and (q.bid and q.ask and q.ask / q.bid >= 0.60)
                current < 0
                and ema < 0
                and q.bid
                and q.ask
                and (abs(q.bid - q.ask) <= 5)
            ):
                # Note: for spreads, we always use only prices we've SEEN SINCE THE QUOTE WAS ADDED and we ignore any
                #       officialy reported "high/low" of the spread (since the "official" high/low data for a spread
                #       is just IBKR reporting if a certain combination traded, so it doesn't represent actual high/low
                #       potential transactions of the legs; and for our own usage, it's easier to see how much a spread
                #       has changed against the highest and lowest value we've seen since we added it).
                # ALSO NOTE: to avoid guessing unseen prices in the middle of a spread, we assume the worst outcome where the low is the highest ask and the high is the smallest bid.
                self.ticker.low = min(self.ticker.low or float("inf"), q.ask)
                self.ticker.high = max(self.ticker.high or float("-inf"), q.bid)

        # if we belong to bags, update the greeks inside the bags
        for bag in self.bags:
            bag.updateGreeks()

        # update EMAs
        # (one problem we have here: when IBKR sometimes refuses to quote spreads, the spread never gets populated in a
        #  ticker update, so the spread will never populate all the metadata (as it has no bid/ask quote and we would
        #  have to generate it synthetically from the legs for additional EMA updating)
        ts = self.ticker.timestamp
        assert ts
        self.ema.update(current, ts)

        name = self.ticker.contract.symbol  # type: ignore
        if isinstance(self.ticker.contract, Future):
            content = ""
            e60 = self.ema[60]
            e300 = self.ema[300]
            diff = current - e300
            d60 = current - e60
            prefix = ""
            if name in {"ES", "RTY"}:
                if diff >= 2 and d60 > 0:
                    if diff > 4:
                        prefix = "MAJOR"
                    content = f"RAPID TICKER UP {name}"
                elif diff <= -2 and d60 < 0:
                    if diff < -4:
                        prefix = "MAJOR"
                    content = f"RAPID TICKER DOWN {name}"
            elif name in {"NQ"}:
                if diff >= 10 and d60 > 0:
                    if diff > 20:
                        prefix = "MAJOR"
                    content = f"RAPID TICKER UP {name}"
                elif diff <= -10 and d60 < 0:
                    if diff < 20:
                        prefix = "MAJOR"
                    content = f"RAPID TICKER DOWN {name}"

            # yes, this is a weird way to use a 'prefix' but it works for information delivery density

            # TODO: fix this logic, it is never triggering
            if content and self.current and self.low:
                atr300 = self.atrs[300].current
                inRangeLow = (
                    "(NEAR LOW ATR)" if (self.current - self.low) <= atr300 else ""
                )
                inRangeHigh = (
                    "(NEAR HIGH ATR)" if (self.high - self.current) <= atr300 else ""
                )
                content = " ".join([content, prefix, inRangeLow, inRangeHigh]).replace(
                    "  ", " "
                )
                self.state.task_create(
                    content,
                    self.state.speak.say(
                        say=content, suppress=60, deadline=time.time() + 5
                    ),
                )

        # check level breaches for alerting
        # compare against ema 30 second to use as the directional bias anchor
        recentHistoryAnchor = round(self.ema[30], 3)
        newer = current
        vw = round(self.vwap, 3)

        # for now, only run VWAP reporting on clients where we have other levels established!
        if self.levels:
            if isinstance(self.ticker.contract, (Future, Index)):
                if recentHistoryAnchor > 1 and vw > 1:
                    # Let's speak VWAP alerts too using local data...
                    if recentHistoryAnchor > vw and newer < vw:
                        # logger.info("down because: {} > {} and {} < {}", recentHistoryAnchor, vw, newer, vw)
                        content = f"{self.name} VW DOWN"
                        self.state.task_create(
                            content,
                            self.state.speak.say(
                                say=content, suppress=60, aux=f" @ {vw:.2f}"
                            ),
                        )
                    elif recentHistoryAnchor < vw and newer > vw:
                        # logger.info("up because: {} < {} and {} > {}", recentHistoryAnchor, vw, newer, vw)
                        content = f"{self.name} VW UP"
                        self.state.task_create(
                            content,
                            self.state.speak.say(
                                say=content, suppress=60, aux=f" @ {vw:.2f}"
                            ),
                        )

        # TODO: also compare against previous daily high and previous daily low
        for duration, breacher in self.levels.items():
            if not breacher.enabled:
                continue

            for level in breacher.levels:
                l = level.level
                if recentHistoryAnchor >= l and newer <= l:
                    # for SMA, we need to say: SMA DURATION SOURCE (e.g. SMA 5 1-day) but we don't want to say "close 1 day 1 day" if duration and lookback are the same
                    addendum = (
                        f", {breacher.durationName}"
                        if breacher.durationName != level.lookbackName
                        else ""
                    )
                    content = f"{self.name} DOWN {level.levelType} {level.lookbackName}{addendum}"
                    self.state.task_create(
                        content, self.state.speak.say(say=content, suppress=60)
                    )
                elif recentHistoryAnchor <= l and newer >= l:
                    addendum = (
                        f", {breacher.durationName}"
                        if breacher.durationName != level.lookbackName
                        else ""
                    )
                    content = f"{self.name} UP {level.levelType} {level.lookbackName}{addendum}"
                    self.state.task_create(
                        content, self.state.speak.say(say=content, suppress=60)
                    )

        self.emaTradeRate.update(self.ticker.tradeRate or 0, ts)
        self.emaVolumeRate.update(self.ticker.volumeRate or 0, ts)

        # update greeks-specific EMAs because why not?
        if g := self.ticker.modelGreeks:
            self.emaIV.update(g.impliedVol, ts)
            self.emaDelta.update(g.delta, ts)
            self.emaVega.update(g.vega, ts)

        # logger.info("[{}] EMAs: {}", self.ticker.contract.localSymbol, self.ema)
        if isinstance(self.ticker.contract, (Future, FuturesOption, Option)):
            name = self.ticker.contract.localSymbol

            # We don't have a clean/concise way of alerting bag names, so avoid for now (bag contracts have no names themselves...)
            if name:
                # fire a special alert if EMA cross is changing directions
                if self.prevDirUp:
                    # if previously up, but now down (fast < slow) , report.
                    # NOTE: we have turned off the DOWN alerts because we typically have equal call and put strikes and there ends up
                    #       being equal reports of DOWN as UP, so we can only report UP (which we tend to care about more anyway).
                    if False and self.ema[60] - self.ema[120] < -0.05:
                        self.prevDirUp = False
                        self.state.task_create(
                            "DOWN",
                            self.state.speak.say(say=f"DOWN {name}", suppress=10),
                        )
                else:
                    if self.ema[60] - self.ema[120] > 0.05:
                        self.prevDirUp = True
                        self.state.task_create(
                            "UP",
                            self.state.speak.say(say=f"UP {name}", suppress=10),
                        )

        # check alert requests for... alerting, I guess.
        baseline = round(self.ema[15], 2)
        prevcompare = 0.0
        if isinstance(self.ticker.contract, (Option, FuturesOption, Bag)):
            # first, check if the delta is sweeping upward
            # (yes, we are only tracking UPWARD motion currently. for downward alerts, just watch an opposite P/C position)
            if mg := self.ticker.modelGreeks:
                curDelta = abs(mg.delta or 0)
                if deltaAlert := self.alertDelta:
                    nextDeltaAlert = curDelta + 0.05
                    if curDelta >= deltaAlert:
                        logger.info(
                            "[{} :: {} :: {:>5}] {:.0f} % from {:>4.02f} -> {:>4.02f} (next: {:>4.02f})",
                            self.ticker.contract.localSymbol,
                            self.name,
                            "DELTA",
                            100 * ((curDelta - deltaAlert) / deltaAlert),
                            deltaAlert,
                            curDelta,
                            nextDeltaAlert,
                        )
                        self.alertDelta = nextDeltaAlert
                else:
                    self.alertDelta = abs(curDelta) + 0.05

            # now alert on the actual premium value being reported by the quote/ticker updates
            for start in [1800]:  # [23400, 3900, 1800, 300]:
                # don't alert if the previous historical value is the same as current
                startval = self.ema[start]
                if startval == prevcompare:
                    continue

                prevcompare = startval

                for r in [0.10]:
                    # TODO: proper digit rounding using instrument db
                    nextalert = round(baseline * (1 + r), 2)
                    if saved := self.alerts[start][r]:
                        (prev, found) = saved
                        if abs(baseline) > abs(found):
                            self.alerts[start][r] = (startval, nextalert)
                            # guard this notice from alerting too much if we add a new quote with unpopluated EMAs
                            if prev > 0.50:
                                logger.info(
                                    "[{} :: {:>5}] {:>5,.0f}% from {:>6.02f}  -> {:>6.02f}  (next: {:>6.02f})",
                                    self.ticker.contract.localSymbol or self.name,
                                    start,
                                    100 * ((baseline - prev) / prev),
                                    prev,
                                    baseline,
                                    nextalert,
                                )
                    else:
                        self.alerts[start][r] = (baseline, round(startval * (1 + r), 2))

    def updateGreeks(self):
        """For bags/spreads, we calculate greeks for the entire spread by combining greeks for each leg.

        Greeks are combined by adding longs and subtracting shorts, all in magnitude of ratio-per-leg.
        """

        final = None
        vwap = 0.0

        # TODO: we should also be checking the "freshness" of the legs quotes so we aren't using stale data.
        #       (e.g. if we remove a quote but still have it as a leg, it isn't getting update anymore...)
        # i.e. composite bag quotes are invalid if all legs don't have live quotes being populated.
        for ratio, leg in self.legs:
            # if any leg is missing, all our values are invalid so do nothing.
            if not leg:
                self.ticker.modelGreeks = None
                self.ticker.vwap = nan
                return

            try:
                if final:
                    final += leg.modelGreeks * ratio
                else:
                    final = leg.modelGreeks * ratio
            except:
                # don't update if any of the leg greeks don't exist.
                self.ticker.modelGreeks = None
                return

        # we set a synthetic modelGreeks object on the bag which normally doesn't exist (but we make it here anyway)
        self.ticker.modelGreeks = final

        # for spread "vwap" we use our live tracked EMA instead of combining VWAP of legs because the VWAP for kinda far OTM
        # legs doesn't reflect underlying price changes over time to because our auto-detected OTM strikes may not have many trades
        # (so: OTM strikes with few trades means the VWAP isn't getting updated over time as price actually moves, but our live tracked
        #      EMA will track the price changes as they float over time (as long as quotes started long ago enough in the past))
        self.ticker.vwap = self.ema[RTH_EMA_VWAP]

        # logger.info("Legs ({}): {}", len(self.legs), pp.pformat(self.legs))
        # logger.info("Updated greeks: {}", final)

    def percentAmtFrom(self, base: float) -> PercentAmount:
        """Return a tuple of current price percent change from 'base' as well as numeric difference between current price and 'base'"""
        c = self.quote().current
        if c and base:
            return (((c - base) / base) * 100, c - base)

        return None, None

    # Note: don't make these properties or else __getattr__ will receive the requests instead of the methods.
    def percentAmtFromHigh(self) -> PercentAmount:
        return self.percentAmtFrom(self.ticker.high)

    def percentAmtFromLow(self) -> PercentAmount:
        return self.percentAmtFrom(self.ticker.low)

    def percentAmtFromOpen(self) -> PercentAmount:
        return self.percentAmtFrom(self.ticker.open)

    def percentAmtFromClose(self) -> PercentAmount:
        return self.percentAmtFrom(self.ticker.close)

    def percentAmtFromVWAP(self) -> PercentAmount:
        # use "real vwap" or, if vwap doesn't exist, use our synthetic 6.5 hour EMA instead.
        return self.percentAmtFrom(self.ticker.vwap or self.ema[RTH_EMA_VWAP])


@dataclass(slots=True)
class CompleteTradeNotification:
    """A wrapper to hold a Trade object and a notifier we can attach to.

    Used by our automated order placement logic to get updates when an _entire_ order
    completes so we can continue scaling in or out of the next steps.
    """

    trade: Trade | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)

    async def orderComplete(self):
        """Wait for the event to trigger then clear it sicne we woke up."""
        await self.event.wait()
        self.event.clear()

    def set(self):
        self.event.set()


@dataclass(slots=True)
class IPosition:
    """A representation of a position with order representation capability for accumulation and distribution."""

    # contract for this ENTIRE POSITION
    # i.e. if this is a spread, .contract here is the BAG for the spread, but Contract in the qtycost and updates will be legs.
    #      otherwise, it will be the same contract for all fields if just representing a single instrument.
    contract: Contract

    # quantity and cost per contract as (qty, cost) for ease of iterating
    qtycost: dict[Contract, tuple[float, float]] = field(default_factory=dict)

    # count how many updates we receive per contract.
    # For spreads, this helps us not make decisions if we have an un-equal number of updates (one leg updated, another not yet)
    # to protect against making decisions with not 100% up-to-date field data.
    updates: dict[Contract, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, contract: Contract, qty: float, cost: float):
        """Replace position for contract given quantity and cost.

        Note: for shorts, qty is input as negative (IBKR default behavior),
            but we turn _cost_ negative for our calculations instead.

        Also note: IBKR reports 'cost' as the full dollar cost, but we store cost as per-share cost.
        """

        self.updates[contract] += 1
        self.qtycost[contract] = (qty, cost)

    @property
    def dataIsSafe(self) -> bool:
        """Verify all contracts have an equal update count.

        If all contracts do not have the same update count, we are in the middle
        of an async update cycle and we can't trust the data completely.

        Also, we must have received at least ONE update for every leg in the contract (if this is a bag)
        for the data to be considered safe/complete.
        """

        # data is okay if we don't have any values populated yet
        if not self.updates:
            return True

        # if bag, we must have all contracts populated with at least one position update
        if isinstance(self.contract, Bag):
            values = list(self.updates.values())
            return all([v == values[0] for v in values]) and len(self.updates) == len(
                self.contract.comboLegs
            )

        # else, not a bag, so we only have one update basically
        assert len(self.updates) == 1

        return True

    @property
    def totalSpend(self) -> float | None:
        """Return the total value spent to acquire this position.

        Note: doesn't account for margin holdings (e.g. if you have 20 point wide credit spread you received $4 credit on, you still are at risk of (20 - 4) margin fill-up.
        """
        if self.dataIsSafe:
            pq = 0.0
            for contract, (qty, price) in self.qtycost.items():
                pq += qty * price

            return pq

        return None

    @property
    def averageCost(self) -> float | None:
        """Return average cost of position in per-share prices.

        Note: for spreads, we just add all the legs together (and this also means we store short legs with negative prices)
        """

        # generate average cost as (position qty * position cost per contract) / (total qty)
        # Note: this won't _exactly_ equal your execution price because IBKR will update your cost basis to be reduced by commissions.
        #       e.g. if you shorted a spread for -$17 credit, you may see your average cost is actually -$16.94 after commissions
        if self.dataIsSafe:
            pq = 0.0
            q = float("inf")
            for contract, (qty, price) in self.qtycost.items():
                pq += qty * price / float(contract.multiplier or 1)

                # quantity is the minimum value of any quantity seen
                # (e.g. for spreads, a butterfly of qty 1:2:1 is butterfly qty 1 for buying/selling,
                #       and a vertical spread qty of +70 long and -70 short is a qty 70 spread (not '-70 + 70 == 0')
                q = min(q, abs(qty))

            return pq / q

        return None

    def closePercent(self, percent: float) -> float | None:
        """Return a price above (+) or below (-) the average cost.

        Can be used to generate take profit price with a positive [0, 1+] percent
        or can be used to generate a stop loss price with a negative [-1, 0] percent.

        Note: the price naturally matches positive percents to ALWAYS be profit and negative
              percents to ALWAYS be loss regardless of the long/short position side.
        """
        if self.dataIsSafe:
            ac = self.averageCost
            assert ac

            return ac + (abs(ac) * percent)

        return None

    def closeCash(self, cash: float) -> float | None:
        """Return a price above (+) or below (-) the average cost.

        Can be used to generate take profit price with a positive price growth
        or can be used to generate a stop loss price with a negative price growth.
        """
        if self.dataIsSafe:
            ac = self.averageCost
            assert ac

            tq = self.totalQty
            assert tq

            # we need to project the average cost price back into a full spent cost basis (ac per share * tq * multiplier) == total spend,
            # then we can add the total profit/lost cash price requested, then divide by (quantity * multiplier) to get the limit price
            # yielding the requested profit/loss limit price given the total spend.
            multiplier = float(self.contract.multiplier or 1)
            actualGrowthAdjuster = tq * multiplier

            # our position 'ac' is postive for longs and negative for shorts/credits, so:
            # POSITIVE CASH on LONG  will MAKE  BIGGER POSITIVE PRICE (+, +) == PROFIT
            # NEGATIVE CASH on LONG  will MAKE SMALLER POSITIVE PRICE (+, -) == LOSS
            # POSITIVE CASH on SHORT will MAKE SMALLER NEGATIVE PRICE (-, +) == PROFIT
            # NEGATIVE CASH on SHORT will MAKE  BIGGER NEGATIVE PRICE (-, -) == LOSS
            targetPrice = ((ac * actualGrowthAdjuster) + cash) / actualGrowthAdjuster

            return targetPrice

        return None

    def limitLoss(self, percent: float) -> float | None:
        if self.dataIsSafe:
            ac = self.averageCost
            assert ac

            closeSide = -1 if ac > 0 else 1
            return ac - (ac * percent) * closeSide

        return None

    @property
    def totalQty(self) -> float | None:
        """Return total quantity for this position.

        Note: quantity for spreads is the smallest quantity of any leg.

        Also note: we only report positive quantities so .isLong is needed for a directional quantity check.
        """

        if self.dataIsSafe:
            return (
                min([abs(x[0]) for x in self.qtycost.values()]) if self.qtycost else 0
            )

        return None

    @property
    def closeQty(self) -> float | None:
        """Return quantity for closing (negative if long, positive if short)"""

        if self.dataIsSafe:
            if ac := self.averageCost:
                closeSide = -1 if ac > 0 else 1
                # return negative quantity if currently long, positive quantity if currently short
                return min([abs(x[0]) for x in self.qtycost.values()]) * closeSide

        return None

    @property
    def isLong(self) -> bool | None:
        if ac := self.averageCost:
            return ac > 0

        return None

    @property
    def name(self) -> str:
        """Generate a text name for this contract we can use to discover the same contract again."""
        return nameForContract(self.contract)

    def percentComplete(self, goal: OrderIntent) -> float | None:
        qty = self.totalQty
        amt = self.totalSpend
        avg = self.averageCost

        # if any of the metadata reporters say we aren't ready yet, we can't do any math.
        if any([x is None for x in (qty, amt, avg)]):
            return None

        # if we have no goal, we are 100% complete because there is nothing to do.
        if not goal.qty:
            return 1

        match goal.qty:
            case qtyTarget if isinstance(qtyTarget, DecimalShares):
                assert qty is not None
                return qty / float(qtyTarget)
            case cashTarget if isinstance(cashTarget, DecimalCash):
                assert amt is not None
                assert avg is not None
                # We estimate the impact of adding half of one more quantity to see
                # if acquiring one more is likely to put us over the goal or not.
                return (amt + (avg / 2)) / float(cashTarget)

        return None

    async def accumulate(self, qty: float | str) -> str:
        if self.isLong:
            assert (isinstance(qty, (float, int, Decimal)) and qty > 0) or (
                isinstance(qty, str) and "-" not in qty
            )
        else:
            assert (isinstance(qty, (float, int, Decimal)) and qty < 0) or (
                isinstance(qty, str) and "-" in qty
            )

        cmd = f"buy '{self.name}' {qty} AF"
        return cmd

    async def distribute(self, qty: float | str) -> str:
        if self.isLong:
            assert (isinstance(qty, (float, int, Decimal)) and qty < 0) or (
                isinstance(qty, str) and "-" in qty
            )
        else:
            assert (isinstance(qty, (float, int, Decimal)) and qty > 0) or (
                isinstance(qty, str) and "-" not in qty
            )

        cmd = f"buy '{self.name}' {qty} AF"
        return cmd


@dataclass(slots=True)
class Bracket:
    profitLimit: Decimal | float | None = None
    lossLimit: Decimal | float | None = None

    # Note: IBKR bracket orders must use common exchange order types (no AF, AS, REL, etc)
    orderProfit: str = "LMT"
    orderLoss: str = "STP LMT"
    lossStop: Decimal | float | None = None

    def __post_init__(self) -> None:
        # if no stop trigger provided, set stop equal to the liss limit
        if not self.lossStop:
            self.lossStop = self.lossLimit


@dataclass(slots=True)
class LadderStep:
    qty: Decimal
    limit: Decimal

    def __post_init__(self) -> None:
        assert isinstance(self.qty, Decimal)
        assert isinstance(self.limit, Decimal)


# Note: DO NOT use slots=True here because we use @cached_property which doesn't work with slots=True
@dataclass(slots=False)
class Ladder:
    """A description of how to order successively higher or lower prices and quantities for an instrument.

    We also call this a "scale" operation for scaling in and out of positions with partial quantities at
    different prices over time.

    One important feature of our ladder/scale operations is the ability to sumbit them all as connected orders
    then attach a final-level average cost stop-out at the end. This allows us to accumulate at a better cost
    basis during mild volatility, but if volatility exceeds our expectations, we stop our position to prevent
    more damage than we expected from occurring.
    """

    # Details Required:
    #  - N Steps
    #    - A step is a combined (qty + limit price)
    #  - Total stop percentage loss acceptable
    #  - After the final step, we calculate a final stop-limit order to cancel this position if it continues
    #    going against our interests.

    # note: percentages here are 0.xx based
    stopPct: Decimal | None = None
    profitPct: Decimal | None = None

    stopAlgo: str = "STP"
    profitAlgo: str = "LMT"

    steps: tuple[LadderStep, ...] = tuple()

    @classmethod
    def fromOrderIntent(self, oi: OrderIntent) -> Ladder:
        """Convert an OrderIntent with embedded scale ladder into an icli ladder format.

        The main reason we don't use OrderIntent ladder directly is the OrderIntent scale
        is just a list of OrderIntent items at different price, quantity levels with no
        collective profit/stop/average cost embedded.

        We could technically create an OrderIntent group scale object too, but also currently
        the placeOrderForContract() doesn't know about OrderIntent objects directly, so we would
        need to refactor placeOrderForContract() to stop using the custom PriceOrQuantity class
        and instead derive all its values from the OrderIntent object (which _is_ better, we just
        haven't had time to properly refactor it yet, so continuing to add extra custom additions
        on top is cleaner for now).
        """
        steps = tuple([LadderStep(qty=o.qty, limit=o.limit) for o in oi.scale])  # type: ignore

        # 0.xx based percentages
        stopPct: Decimal | None = None
        profitPct: Decimal | None = None

        if isinstance(oi.bracketProfit, DecimalPercent):
            profitPct = oi.bracketProfit / D100
            if oi.isShort:
                profitPct = -profitPct

        if isinstance(oi.bracketLoss, DecimalPercent):
            stopPct = oi.bracketLoss / D100
            if oi.isShort:
                stopPct = -stopPct

        return Ladder(
            profitPct=profitPct,
            stopPct=stopPct,
            steps=steps,
            profitAlgo=ALGOMAP.get(oi.bracketProfitAlgo, None),  # type: ignore
            stopAlgo=ALGOMAP.get(oi.bracketLossAlgo, None),  # type: ignore
        )

    def __bool__(self) -> bool:
        """Ladder only exists if it has order steps inside"""
        return len(self.steps) > 0

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        yield from self.steps

    @cached_property
    def qty(self) -> Decimal:
        return sum([x.qty for x in self.steps])  # type: ignore

    @cached_property
    def avgPrice(self) -> Decimal:
        tpq = Decimal()
        tq = Decimal()
        for x in self.steps:
            tpq += x.qty * x.limit
            tq += x.qty

        return tpq / (tq or 1)

    @cached_property
    def stop(self) -> Decimal | None:
        if not self.stopPct:
            return None

        avg = self.avgPrice

        # calculate average stop loss condition which is LOWER for longs but HIGHER for shorts
        return avg - (avg * self.stopPct)

    @cached_property
    def profit(self) -> Decimal | None:
        if not self.profitPct:
            return None

        avg = self.avgPrice

        # calculate average profit condition which is HIGHER for longs but LOWER for shorts
        return avg + (avg * self.profitPct)


def convert_futures_code(code: str):
    """Convert a futures-date-format into IBKR date format.

    So input like 'Z3' becomes 202312.
    """

    assert (
        len(code) == 2 and code[-1].isdigit()
    ), f"Futures codes are two characters like F3 for January 2023, but we got: {code}"

    # Mapping for month codes as per future contracts
    try:
        month_code = FUTS_MONTH_MAPPING[code[0].upper()]
    except KeyError:
        raise ValueError("Invalid month code in futures contract")

    # our math accounts for any numbers PREVIOUS to this year are for the NEXT decade,
    # while numbers FOWARD from here are for the current decade.
    current_year = datetime.datetime.now().year
    year_decade_start = current_year - current_year % 10
    year = year_decade_start + int(code[1])

    return str(year) + month_code


def find_nearest(lst, target):
    """
    Finds the nearest number in a sorted list to the given target.

    If there is an exact match, that number is returned. Otherwise, it returns the number with the smallest
    numerical difference from the target within the list.

    Args:
        lst (list): A sorted list of numbers.
        target (int): The number for which to find the nearest value in the list.

    Returns:
        The nearest index to `target` in `lst`.

    Bascially: using ONLY bisection causes rounding problems because if a query is just 0.0001 more than a value
               in the array, then it picks the NEXT HIGHEST value, but we don't want that, we want the NEAREST value
               which minimizes the difference between the input value and all values in the list.

               So, instead of just "bisect and use" we do the bisect then compare the numerical difference between
               the current element and the next element to decide whether to round down or up from the current value.
    """

    # Get the index where the value should be inserted (rounded down)
    idx = bisect.bisect_left(lst, target) - 1

    size = len(lst)

    # If the difference to the current element is less than or equal to the difference to the next element
    try:
        # this is equivalent to MATCHING or ROUNDING DOWN
        if idx >= 0 and abs(target - lst[idx]) <= abs(target - lst[idx + 1]):
            return idx
    except:
        # if list[idx + 1] doesn't exist (beyond the list) just fall through and we'll return "size - 1" which is the maximum position.
        pass

    # If we need to round up, return the next index in the list (or the final element if we've reached beyond the end of the list)
    return idx + 1 if idx < size - 1 else size - 1


def nameForContract(contract: Contract, cdb: dict[int, Contract] | None = None) -> str:
    """Generate a text description for a contract we can re-parse into a contract again.

    The goal here is to provide a more user-readable contract description for logs or metadata details
    than just printing the underlying contract ids everywhere.
    """

    if isinstance(contract, Option) or contract.secType == "OPT":
        return contract.localSymbol.replace(" ", "")

    if contract.secType in {"FOP", "EC"}:
        # Note: this could also technically be just "/" + localSymbol.replace(" ", "") becase we can read futures option syntax now too
        return f"/{contract.symbol}{contract.lastTradeDateOrContractMonth[2:]}{contract.right}{int(contract.strike * 1000):08}-{contract.tradingClass}"

    if isinstance(contract, Bag):
        result = []

        # TODO: if we want more descriptive names (instead of contract ids) this needs to be attached to a place where we can read cached contracts...
        if cdb:
            # if we have the contract db available, we can generate more meaningful details
            for leg in contract.comboLegs:
                foundLeg = cdb.get(leg.conId)
                result.extend(
                    [
                        leg.action,
                        str(leg.ratio),
                        nameForContract(foundLeg) if foundLeg else str(leg.conId),
                    ]
                )
        else:
            for leg in contract.comboLegs:
                result.extend([leg.action, str(leg.ratio), str(leg.conId)])

        return " ".join(result)

    if isinstance(contract, Future):
        return f"/{contract.localSymbol}"

    if isinstance(contract, Stock):
        return contract.localSymbol

    if isinstance(contract, Index):
        return f"I:{contract.localSymbol}"

    if isinstance(contract, Forex):
        return f"F:{contract.localSymbol}"

    if isinstance(contract, Crypto):
        return f"C:{contract.localSymbol}"

    assert None, f"Unexpected contract for name creation? {contract=}"


def contractForName(sym, exchange="SMART", currency="USD") -> Contract:
    """Convert a single text symbol data format into an ib_insync contract."""

    sym = sym.upper()

    contract: Contract

    # TODO: how to specify warrants/equity options/future options/spreads/bonds/tbills/etc?
    if sym.isnumeric() and len(sym) > 3:
        # Allow exact contract IDs to generate an abstract contract we then qualify to a more concrete type.
        # (also only accept numbers longer than 3 digits so a typo of positions like :13 as 13 doesn't become a contract id by mistake)
        contract = Contract(conId=int(sym))
    elif sym.startswith("/") or sym.startswith(","):
        # (in some places we use ',' as a futures prefix instead of '/' if we had to serialize out a symbol name as a file...)
        sym = sym[1:]

        # Check if this symbol is directly in our futures lookup map (or two minus the end of the end is a month/year indicator)...
        inFutureMap = sym in FUTS_EXCHANGE or (
            len(sym) >= 4 and sym[:-2] in FUTS_EXCHANGE
        )

        # first check if this is a CME syntax for FOP at lengths 9, 10, 13 (/EWZ4P4000) (/E4AU4C4700) or EC (/ECESZ431P4000)
        # Also consider: /GCZ4C2665 or /GCZ4C2665-EC or /ECGCZ4C2665 or /ECBTCZ4C63500 or /BTCZ4C63500-EC
        if (not inFutureMap) and (9 <= len(sym) <= 13):
            if "-" in sym:
                sympart, tradingClass = sym.split("-")
            else:
                sympart = sym
                tradingClass = None

            halfsearch = len(sympart) // 2 - 1
            try:
                mid = sympart.index("C", halfsearch)
            except:
                mid = sympart.index("P", halfsearch)

            strike = sympart[mid:]
            symbol = sympart[: -len(sympart) + mid]

            if not tradingClass:
                tradingClass = symbol[:-2]

            localSymbol = f"{symbol} {strike}"

            if symbol.startswith("EC"):
                # remove EC prefix and date code
                exchangeSymbol = symbol[2:-2]
            else:
                exchangeSymbol = tradingClass

            fxchg = FUTS_EXCHANGE.get(exchangeSymbol)

            contract = FuturesOption(
                exchange=fxchg.exchange if fxchg else "CME",
                localSymbol=localSymbol,
                tradingClass=tradingClass,
            )
        elif len(sym) > 15:
            # else, use our custom hack format allowing CME futures options to look like OCC options syntax
            if "-" in sym:
                sym, tradingClass = sym.split("-")
            else:
                tradingClass = ""

            # Is Future Option! FOP!
            symbol = sym[:-15]

            body = sym[-15:]
            date = "20" + body[:6]
            right = body[-9]  # 'C'

            if right not in {"C", "P"}:
                raise Exception(f"Invalid option format right: {right} in {sym}")

            price = int(body[-8:])  # 320000 (leading 0s get trimmed automatically)
            strike = price / 1000  # 320.0

            # fix up if has date code embedded in the symbol
            if symbol[-1].isdigit():
                symbol = symbol[:-2]

            fxchg = FUTS_EXCHANGE[symbol]
            contract = FuturesOption(
                currency=currency,
                symbol=fxchg.symbol,
                exchange=fxchg.exchange,
                strike=strike,
                right=right,
                lastTradeDateOrContractMonth=date,
                tradingClass=tradingClass,
            )
        else:
            # else, is regular future (or quote-like thing we treat as a future)

            # our symbol lookup table is the unqualified contract name like "ES" but
            # when trading, the month and year gets added like "ESZ3", so if we have
            # a symbol ending in a digit here, remove the "expiration/year" designation
            # from the string to lookup the actual name.
            dateForContract = FUT_EXP
            if sym[-1].isdigit() and sym[-2] in FUTS_MONTH_MAPPING:
                fullsym = sym
                sym = sym[:-2]

                # if we have an EXACT date syntax requested, populate it instead of the default "current next main future expiration quarter"
                dateForContract = convert_futures_code(fullsym[-2:])

            try:
                fxchg = FUTS_EXCHANGE[sym]
            except:
                logger.error("[{}] Symbol not in our futures database mapping!", sym)
                raise ValueError(f"Unknown future mapping requested: {sym}")

            if dateForContract == FUT_EXP and fxchg.name.endswith("Yield"):
                # "Yield" products expire MONTHLY and not quarterly, so do big end-of-month smash here
                # (if you want a *specific* forward month (usually only listed current and next 2 months at once), you
                # can use the more common futures codes like /10YN4 to mean July 2024 etc. By default you'll get THIS MONTH expiry.
                # TODO: technically the "is Yield type" should be a property of the futures mapping instead of this
                #       more hacky "if name ends in Yield, it's a yield quote, so use monthly expirations..."
                now = datetime.datetime.now().date()
                dateForContract = f"{now.year}{now.month:02}"

            isweeklyvix = sym[:2] == "VX"

            # We need some extra consideration for populating weekly vix contracts because
            # we reference them by local symbol for trading like /VX17, but they are still "symbol VIX" and
            # the _trading class_ is the input symbol requested (VX17).
            # Also, to run this properly, you must manually override the date spec with futures month/year symbols
            # so you bind the correct month/year to the weekly expiration like /VX17J5 (could be more automated, but isn't yet).
            # See the "Contract Expirations" header here for more details: https://www.cboe.com/tradable_products/vix/vix_futures/specifications/
            if isweeklyvix:
                contract = Future(
                    currency=currency,
                    symbol=fxchg.symbol,
                    exchange=fxchg.exchange,
                    lastTradeDateOrContractMonth=dateForContract,
                    tradingClass=sym,
                )
            else:
                contract = Future(
                    currency=currency,
                    symbol=fxchg.symbol,
                    exchange=fxchg.exchange,
                    # if it looks like our symbol ends in a futures date code, convert the futures
                    # date code to IBKR date format. else, use our default continuous next-expiry futures calculation.
                    lastTradeDateOrContractMonth=dateForContract,
                    tradingClass="",
                )
    elif len(sym) > 15:
        # looks like: COIN210430C00320000
        symbol = sym[:-15]  # COIN
        body = sym[-15:]  # 210430C00320000

        # Note: Not year 2100+ compliant!
        # 20 + YY + MM + DD
        date = "20" + body[:6]

        right = body[-9]  # 'C'

        if right not in {"C", "P"}:
            raise Exception(f"Invalid option format right: {right} in {sym}")

        price = int(body[-8:])  # 320000 (leading 0s get trimmed automatically)
        strike = price / 1000  # 320.0

        contract = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=date,
            strike=strike,
            right=right,
            exchange=exchange,
            currency=currency,
            # also pass in tradingClass so options like SPXW220915P03850000 work
            # directly instead of having IBKR guess if we want SPX or SPXW.
            # for all other symbols the underlying trading class doesn't alter
            # behavior (and we don't allow users to specify extra classes yet
            # like if you want to trade on fragemented options chains after
            # reverse splits, etc).
            tradingClass=symbol,
        )
    else:
        # if symbol has a : we are namespacing by type:
        #   - W: - warrant
        #   - C: - crypto
        #   - F: - forex
        #   - B: - bond
        #   - S: - stock (or no contract namespace)
        #   - I: - an index value (VIX, VIN, SPX, etc)
        #   - K: - a direct IBKR contract id to populate into a contract
        # Note: futures and future options are prefixed with /
        #       equity options are the full OCC symbol with no prefix
        namespaceParts = sym.split(":")
        if len(namespaceParts) > 1:
            contractNamespace, symbol = namespaceParts
            if contractNamespace == "W":
                # TODO: needs option-like strike, right, multiplier, contract date spec too
                contract = Warrant(
                    symbol=symbol, exchange=exchange, currency=currency, right="C"
                )
                # Requires all the details like:
                # contract = Warrant(conId=504262528, symbol='BGRY', lastTradeDateOrContractMonth='20261210', strike=11.5, right='C', multiplier='1', primaryExchange='NASDAQ', currency='USD', localSymbol='BGRYW', tradingClass='BGRY')
            elif contractNamespace == "C":
                contract = Crypto(symbol=symbol, exchange="PAXOS", currency=currency)
            elif contractNamespace == "B":
                contract = Bond(symbol=symbol, exchange=exchange, currency=currency)
            elif contractNamespace == "S":
                contract = Stock(symbol, exchange, currency)
            elif contractNamespace == "I":
                # this appears to work fine without specifying the full Index(symbol, exchange) format
                contract = Index(symbol)
            elif contractNamespace == "F":
                # things like F:GBPUSD F:EURUSD (but not F:JPYUSD or F:RMBUSD shrug)
                # also remember C: is CRYPTO not "CURRENCY," so currencies are F for FOREX
                contract = Forex(symbol)
            elif contractNamespace == "CFD":
                # things like CFD:XAUUSD
                contract = CFD(symbol)
            elif contractNamespace == "K":
                contract = Contract(conId=int(symbol))
            else:
                logger.error("Invalid contract type: {}", contractNamespace)
                raise ValueError(
                    f"Invalid contract type requested: {contractNamespace}"
                )
        else:
            # TODO: warrants, bonds, bills, etc
            contract = Stock(sym, exchange, currency)

    return contract


def contractToSymbolDescriptor(contract) -> str:
    """Extracts the class name of a contract to return className-Symbol globally unique string"""

    # We need the input contract request to generate a strong enough primary key where it doesn't conflict
    # with other contracts. So we can't just do "Class-Symbol" because every option would be e.g. "Option-MSFT".
    # NOTE: just remember to only include "user lookup fields" in the primary key. The user isn't populating things like
    #       'tradingClass' so we don't want to use it in the primary key even though it does get populated after the qualify.
    # The cache storage is looked up using UNQUALIFIED contracts then stored using the QUALIFIED contracts.
    # Also note: we use 'contract.secType' instead of 'contract.__class__.__name__' because some contracts don't have exact
    #            class types, but we end up populating similar class types with different secTypes underneath.
    parts = (
        contract.secType,
        contract.localSymbol,
        contract.symbol,
        contract.lastTradeDateOrContractMonth or "NoDate",
        contract.right or "NoRight",
        str(float(contract.strike or 0) or "NoStrike"),
        contract.tradingClass or "NoTradingClass",
    )
    return "-".join(parts)


def contractFromTypeId(contractType: str, conId: int) -> Contract:
    """Consume a previously extract contract class name and conId to generate a new proper concrete subclass of Contract"""
    match contractType:
        case "Bag":
            return Bag(conId=conId)
        case "Bond":
            return Bond(conId=conId)
        case "CFD":
            return CFD(conId=conId)
        case "Commodity":
            return Commodity(conId=conId)
        case "ContFuture":
            return ContFuture(conId=conId)
        case "Crypto":
            return Crypto(conId=conId)
        case "Forex":
            return Forex(conId=conId)
        case "Future":
            return Future(conId=conId)
        case "FuturesOption":
            return FuturesOption(conId=conId)
        case "Index":
            return Index(conId=conId)
        case "MutualFund":
            return MutualFund(conId=conId)
        case "Option":
            return Option(conId=conId)
        case "Stock":
            return Stock(conId=conId)
        case "Warrant":
            return Warrant(conId=conId)
        case _:
            raise ValueError(f"Unsupported contract type: {contractType}")


def contractFromSymbolDescriptor(contractType: str, symbol: str):
    match contractType:
        case "Bag":
            return Bag(symbol=symbol)
        case "Bond":
            return Bond(symbol=symbol)
        case "CFD":
            return CFD(symbol=symbol)
        case "Commodity":
            return Commodity(symbol=symbol)
        case "ContFuture":
            return ContFuture(symbol=symbol)
        case "Crypto":
            return Crypto(symbol=symbol)
        case "Forex":
            return Forex(symbol=symbol)
        case "Future":
            return Future(symbol=symbol)
        case "FuturesOption":
            return FuturesOption(symbol=symbol)
        case "Index":
            return Index(symbol=symbol)
        case "MutualFund":
            return MutualFund(symbol=symbol)
        case "Option":
            return Option(symbol=symbol)
        case "Stock":
            return Stock(symbol=symbol)
        case "Warrant":
            return Warrant(symbol=symbol)
        case _:
            raise ValueError(f"Unsupported contract type: {contractType}")


def tickFieldsForContract(contract) -> str:
    # Available fields from:
    # https://interactivebrokers.github.io/tws-api/tick_types.html
    # NOTE: the number to use here is the 'Generic tick required' number and NOT the 'Tick id' number.
    extraFields = [
        # start with VWAP and volume data requested everywhere
        233,
        # also add "volume per minute"
        295,
        # also add "trades per minute"
        294,
    ]

    # There is also "fundamentals" as tick 258 but it returns things like this which isn't useful for us because
    # it's just reporting on historical financial reports:
    #     fundamentalRatios=FundamentalRatios(TTMNPMGN=35.20226, NLOW=274.38, TTMPRCFPS=18.29654, TTMGROSMGN=81.49056, TTMCFSHR=25.00352, QCURRATIO=2.83036, TTMREV=149783, TTMINVTURN=nan, TTMOPMGN=39.25479, TTMPR2REV=8.03502, AEPSNORM=16.18888, TTMNIPEREM=783264.3, EPSCHNGYR=82.25327, TTMPRFCFPS=25.55114, TTMRECTURN=11.08847, TTMPTMGN=40.1354, QCSHPS=22.92933, TTMFCF=47102, LATESTADATE='2024-06-30', APTMGNPCT=35.15737, AEBTNORM=50880, TTMNIAC=52727, NetDebt_I=-39691, PRYTDPCTR=23.60872, TTMEBITD=73664, AFEEPSNTM=22.401, PR2TANBK=8.90664, EPSTRENDGR=14.79464, QTOTD2EQ=11.73045, TTMFCFSHR=17.9044, QBVPS=61.88827, NPRICE=475.73, YLD5YAVG=nan, PR13WKPCT=2.15813, PR52WKPCT=53.10076, REVTRENDGR=19.29375, AROAPCT=19.10341, TTMEPSXCLX=20.0446, QTANBVPS=53.34583, PRICE2BK=7.68692, MKTCAP=1203510, TTMPAYRAT=4.84951, TTMINTCOV=nan, TTMREVCHG=24.27752, TTMROAPCT=24.13544, TTMROEPCT=36.26391, TTMREVPERE=2225040, APENORM=29.38622, TTMROIPCT=27.75098, REVCHNGYR=22.10069, CURRENCY='USD', DIVGRPCT=nan, TTMEPSCHG=136.651, PEEXCLXOR=23.73357, QQUICKRATI=nan, TTMREVPS=56.93547, BETA=1.17755, TTMEBT=60116, ADIV5YAVG=nan, ANIACNORM=42560.56, PR1WKPCT=2.15155, QLTD2EQ=11.73045, NHIG=542.81, PR4WKPCT=-10.12431)

    if isinstance(contract, Stock):
        # 104:
        # "The 30-day historical volatility (currently for stocks)."
        # 106:
        # "The IB 30-day volatility is the at-market volatility estimated
        #  for a maturity thirty calendar days forward of the current trading
        #  day, and is based on option prices from two consecutive expiration
        #  months."
        # 236:
        # "Number of shares available to short"
        # "Shortable: < 1.5, not availabe
        #             > 1.5, if shares can be located
        #             > 2.5, enough shares are available (>= 1k)"
        # 595: Stock volume averaged over 3 minutes, 5 minutes, 10 minutes.
        extraFields += [104, 106, 236, 595]

    # yeah, the API wants a CSV for the tick list. sigh.
    tickFields = ",".join([str(x) for x in extraFields])

    # logger.info("[{}] Sending fields: {}", contract, tickFields)
    return tickFields


def parseContractOptionFields(contract, d):
    # logger.info("contract is: {}", o.contract)
    if isinstance(contract, (Warrant, Option, FuturesOption)):
        try:
            d["date"] = dateutil.parser.parse(
                contract.lastTradeDateOrContractMonth
            ).date()  # type: ignore
        except:
            logger.error("Row didn't have a good date? {}", contract)
            return

        d["strike"] = contract.strike
        d["PC"] = contract.right
    else:
        # populate columns for non-contracts/warrants too so the final
        # column-order generator still works.
        d["date"] = None
        d["strike"] = None
        d["PC"] = None


def sortLocalSymbol(s):
    """Given tuple of (occ date/right/strike, symbol) return
    tuple of (occ date, symbol)"""

    return (s[0][:6], s[1])


def portSort(p):
    """sort portfolioItem 'p' by symbol if stock or by (expiration, symbol) if option"""
    s = tuple(reversed(p.contract.localSymbol.split()))
    if len(s) == 1:
        return s

    # if option, sort by DATE then SYMBOL (i.e. don't sort by (date, type, strike) complex
    return sortLocalSymbol(s)


def tradeOrderCmp(o):
    """Return the sort key for a trade representing a live order.

    The goal is to sort by:
        - BUY / SELL
        - DATE (if has date, expiration, option, warrant, etc)
        - SYMBOL

    Sorting is also flexible where if no date is available, the sort still works fine.
    """

    # Sort all options by expiration first then symbol
    # (no impact on symbols without expirations)
    useSym = o.contract.symbol
    useName = useSym
    useKey = o.contract.localSymbol.split()
    useDate = -1

    # logger.info("Using to sort: {}", o)

    if useKey:
        useName = useKey[0]
        if len(useKey) == 2:
            useDate = useKey[1]
        else:
            # the 'localSymbol' date is 2 digit years while the 'lastTradeDateOrContractMonth' is
            # four digit years, so to compare, strip the leading '20' from LTDOCM
            useDate = o.contract.lastTradeDateOrContractMonth[2:]

    # logger.info("Generated sort key: {}", (useDate, useSym, useName))

    return (o.log[-1].status, str(useDate), str(useSym), str(useName))


def boundsByPercentDifference(mid: float, percent: float) -> tuple[float, float]:
    """Returns the lower and upper percentage differences from 'mid'.

    Percentage is given as a full decimal percentage.
    Example: 0.25% must be provided as 0.0025"""
    # Obviously doesn't work for mid == 0 or percent == 2, but we shouldn't
    # encounter those values under normal usage.

    # This is just the percentage difference between two prices equation
    # re-solved for a and b from: (a - b) / ((a + b) / 2) = percent difference
    lower = -(mid * (percent - 2)) / (percent + 2)
    upper = -(mid * (percent + 2)) / (percent - 2)
    return (lower, upper)


def strFromPositionRow(o):
    """Return string describing an order (for quick review when canceling orders).

    As always, 'averageCost' field is for some reason the cost-per-contract field while
    'marketPrice' is the cost per share, so we manually convert it into the expected
    cost-per-share average cost for display."""

    useAvgCost = o.averageCost / float(o.contract.multiplier or 1)
    digits = 2  # TODO: move this so we can read digits

    return f"{o.contract.localSymbol} :: {o.contract.secType} {o.position:,.{digits}f} MKT:{o.marketPrice:,.{digits}f} CB:{useAvgCost:,.{digits}f} :: {o.contract.conId}"


def isset(x: float | Decimal | None) -> bool:
    """Sadly, IBKR/ib_insync API uses FLOAT_MAX to mean "number is unset" instead of
    letting numeric fields be Optional[float] where we could just check for None.

    So we have to directly compare against another value to see if a returned float
    is a _set_ value or just a placeholder for the default value. le sigh."""

    # the round hack is because sometimes we convert the floats to 2 digits which makes them rather... smaller
    return (
        (x is not None)
        and (x != ib_async.util.UNSET_DOUBLE)
        and (x != round(ib_async.util.UNSET_DOUBLE, 2))
    )


@dataclass
class Q:
    """Self-asking series of prompts."""

    name: str = ""
    msg: str = ""
    choices: Sequence[str | Choice] | None = None
    value: str = field(default_factory=str)

    def __post_init__(self):
        # Allow flexiblity with assigning msg/name if they are just the same
        if not self.msg:
            self.msg = self.name

        if not self.name:
            self.name = self.msg

    def ask(self, **kwargs):
        """Prompt user based on types provided."""
        if self.choices:
            # Note: no kwargs on .select() because .select()
            #       is injecting its own bottom_toolbar for error reporting,
            #       even though it never seems to use it?
            #       See: questionary/prompts/common.py create_inquier_layout()
            return questionary.select(
                message=self.msg,
                choices=self.choices,
                use_indicator=True,
                use_shortcuts=True,
                use_arrow_keys=True,
                use_jk_keys=False,
                # **kwargs,
            ).ask_async()

        return questionary.text(self.msg, default=self.value, **kwargs).ask_async()


@dataclass
class CB:
    """Self-asking series of prompts."""

    name: str = ""
    msg: str = ""
    choices: Sequence[Choice] | None = None

    def __post_init__(self):
        # Allow flexiblity with assigning msg/name if they are just the same
        if not self.msg:
            self.msg = self.name

        if not self.name:
            self.name = self.msg

    def ask(self, **kwargs):
        """Prompt user based on types provided."""
        if self.choices:
            # Note: no kwargs on .select() because .select()
            #       is injecting its own bottom_toolbar for error reporting,
            #       even though it never seems to use it?
            #       See: questionary/prompts/common.py create_inquier_layout()
            return questionary.checkbox(
                message=self.msg,
                choices=self.choices,
                use_jk_keys=False,
                # **kwargs,
            ).ask_async()

        return questionary.text(self.msg, **kwargs).ask_async()


# Note: we use a custom key here instead of just letting the
#       contract hash itself (hash(contract)) because the contract is hashed only
#       by id, but sometimes we have un-populated contracts like Contract(id=X) which
#       then generates an invalid lookup key result because the name is missing.
#       So we want to cache contracts based on their full details so we return different results
#       for fully qualified contract details versus partial contract details.
# TODO: though, if we make Contract types immutable, then we could just do id(contract) as a key.
@cached(cache={}, key=lambda x: x)  # contractToSymbolDescriptor(x))
def lookupKey(contract):
    """Given a contract, return something we can use as a lookup key.

    Needs some tricks here because spreads don't have a built-in
    one dimensional representation."""

    # if this is a spread, there's no single symbol to use as an identifier, so generate a synthetic description instead
    if isinstance(contract, Bag):
        # Generate a custom tuple representation we can use as immutable dict keys.
        # Only the ratio, side/action, and contract id matters when defining collective spread definitions.
        return tuple(
            [
                (b.ratio, b.action, b.conId)
                for b in sorted(
                    contract.comboLegs, key=lambda x: (x.ratio, x.action, x.conId)
                )
            ]
        )

    # else, is not a spread so we can use regular in-contract symbols
    if contract.localSymbol:
        return contract.localSymbol.replace(" ", "")

    # else, if a regular symbol but DOESN'T have a .localSymbol (means
    # we added the quote from a contract without first qualifying it,
    # which works, it's just missing extra details)
    if contract.symbol:
        return contract.symbol

    logger.error("Your contract doesn't have a symbol? Bad contract: {}", contract)

    return None


@dataclass(slots=True)
class PriceOrQuantity:
    """A wrapper/box to allow users to provide price OR quantity using one variable based on input syntax.

    e.g. "$300" is ... price $300... while "300" is quantity 300.
    """

    value: str | int | float | Decimal
    qty: float | int | Decimal = field(init=False)

    is_quantity: bool = False
    is_money: bool = False

    is_long: bool = True

    # hack for now. Fix in a better place eventually.
    exchange: str | None = "SMART"

    def __post_init__(self) -> None:
        # TODO: different currency support?

        if isinstance(self.value, (int, float, Decimal)):
            assert (
                self.is_quantity ^ self.is_money
            ), f"If you provided a direct quantity, you must enable only one of quantity or money, but got: {self}"

            self.qty = self.value

            if self.qty < 0:
                self.is_long = False

                # we don't deal in negative quantities here because IBKR sells have a sell action.
                # negative quantities only happen for prices of credit spreads because those are all "BUY [negative money]"
                self.qty = abs(self.qty)  # type: ignore
        else:
            # else, input is a string and we auto-detect money-vs-quantity depending on if the
            # string value starts with '$' (is money value) or not (is direct quantity)

            assert isinstance(self.value, str)

            # allow numbers to use '_' or ',' for any digit breaks
            self.value = self.value.replace("_", "").replace(",", "")

            # if we have a negative sign in the value, consider it a short.
            # allow: -10 -$10 and $-10 to all activate the short detector.
            if self.value.startswith("-") or self.value.startswith("$-"):
                self.is_long = False
                self.value = self.value.replace("-", "")

            if self.value[0] == "$":
                self.is_money = True
                self.qty = float(self.value[1:])
            else:
                self.qty = float(self.value)
                self.is_quantity = True

        # if there's no fractional component, use integer quantity directly
        iqty = int(self.qty)
        if self.qty == iqty:
            self.qty = iqty

    def __repr__(self) -> str:
        if self.is_quantity:
            return f"{self.qty:,.2f}"

        return locale.currency(self.qty, grouping=True)


async def getExpirationsFromTradier(symbol: str):
    """Fetch option chain expirations and strikes from Tradier.

    I'm tired of IBKR data causing minutes of blocking delays during operations, so let's just use other providers instead.
    """

    # TODO: should we make `token` a parameter instead?
    token = os.getenv("TRADIER_KEY")
    if not token:
        raise Exception("Tradier Token Needed for Tradier Data Fetching!")

    # https://documentation.tradier.com/brokerage-api/markets/get-options-expirations
    async with httpx.AsyncClient() as client:
        got = await client.get(
            "https://api.tradier.com/v1/markets/options/expirations",
            params={
                "symbol": symbol,
                "includeAllRoots": "true",
                "strikes": "true",
                "contractSize": "true",
                "expirationType": "true",
            },
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )

    if not got:
        return None

    found = got.json()

    # API returns map of {expiration: strikes}
    #               e.g. {"20240816": [100, 101, 102, 103, 104, ...], ...}
    result = {}

    # convert returned tradier JSON in to a format compat with how we store IBKR data
    expirations = found["expirations"]

    if not expirations:
        return None

    for date in expirations["expiration"]:
        strikes = date["strikes"]["strike"]

        # fix awful tradier data formats where if only one element exists, it is a scalar and not a list
        # like all the other fields.
        if not isinstance(strikes, list):
            strikes = [strikes]

        # tradier sometimes has bad data where they list only 1-2 strikes for an expiration date?
        # kinda odd, so just skip those dates entirely.
        if len(strikes) < 5:
            continue

        # convert date to IBKR format with no dashes as YYYYMMDD
        result[date["date"].replace("-", "")] = strikes

    return result


def split_commands(text):
    """A helper for splitting in-quote commands delimited by semicolons.

    We can't just split the whole string by semicolons because we have to respect the string boundaries
    if there are quoted elements, so we just get to iterate the entire string character by character. yay.
    """
    # Remove comments
    text = re.sub(r"\s+#.*", "", text).strip()

    # Initialize variables
    commands = []
    current_command = ""
    in_quotes = False
    escape_next = False

    for char in text:
        if escape_next:
            current_command += char
            escape_next = False
        elif char == "\\":
            current_command += char
            escape_next = True
        elif char == '"':
            current_command += char
            in_quotes = not in_quotes
        elif char == ";" and not in_quotes:
            commands.append(current_command.strip())
            current_command = ""
        else:
            current_command += char

    if current_command:
        commands.append(current_command.strip())

    return commands


@dataclass(slots=True)
class AlgoBinder:
    """Consume an external data feed and save results to a dot-addressable dict hierarchy."""

    url: str | None = "ws://127.0.0.1:4442/bars"

    # where we save our results for reading
    data: Any = field(default_factory=lambda: defaultdict(dict))

    # active websocket connection (if any)
    activeWS: Any | None = None

    def read(self, depth: str) -> Any | None:
        """Read something from the saved data in dotted string format.

        e.g. Requesting read("AAPL.30.lb.vwap.5.sma") would return the value
             for: {"AAPL": {"30": {"lb": {"vwap": {"5": {"sma": VALUE}}}}}}

        Then for detecting 5/10 crossovers you could do math on:
             read("AAPL.30.lb.vwap.5.sma") > read("AAPL.30.lb.vwap.10.sma")

        Due to how we receive JSON dicts, we expect ALL key types to be strings,
        so just splitting the input string on dots should work (if all your keys exist).

        Of course, your field names must not include dots in keys anywhere or the entire lookup will break.

        If you provide an invalid or non-existing path, the result is None because the depth will fail.
        """

        val = self.data
        for level in depth.split("."):
            if (val := val.get(level)) is None:
                break

        return val

    async def datafeed(self) -> None:
        """Generator for returning collecting external API results locally.

        We assume the external API is returning an N-level nested dict with layout like:
            {
                symbol1: {
                            duration1: {field1: ...},
                            duration2: {field1: ...},
                            ...
                         },
                symbol2: {
                            duration1: {field1: ...},
                            duration2: {field1: ...},
                            ...
                         },
                ...
            }
        The inner (second-level) 'duration' dicts don't need to be populated on
        every new data update because we just replace each 'duration' dict under
        each symbol on every update (i.e. we don't replace each _full symbol_ on
        each update, but rather we just replace _individual_ 2nd level "duration"
        keys inside each symbol dict on each update).
        """

        assert self.url
        logger.info("[Algo Binder] Connecting to: {}", self.url)

        try:
            # this async context manager automatically handles reconnecting when required
            async for ws in websockets.connect(
                self.url,
                ping_interval=10,
                ping_timeout=30,
                open_timeout=2,
                close_timeout=1,
                max_queue=2**32,
                # big read limit... some of our inbound data is big.
                read_limit=1024 * 2**20,
                # Set max size to "unlimited" (via None) because our initial data size can be 5+ MB for a full symbol+duration+algo+lookback state.
                max_size=None,
                compression=None,
                user_agent_header=None,
            ):
                self.activeWS = ws
                logger.info("[Algo Binder :: {}] Connected!", self.url)

                try:
                    # logger.info("Waiting for WS message...")
                    async for msg in ws:
                        # logger.info("Got msg: {:,} bytes", len(msg))
                        for symbol, durations in ourjson.loads(msg).items():
                            # Currently we expect each symbol to have about 8 first-level
                            # durations representing bar sizes in seconds with something like:
                            # 15, 35, 55, 90, 180, 300, 900, 1800
                            # Also note: we do not REPLACE all symbol durations each update, because
                            #            durations are not updated all at once. One update packet
                            #            may have durations 15, 35 while another may have 90 and 180,
                            #            so we must MERGE new durations into the symbol for each update.
                            # logger.info("Got data: {} :: {}", symbol, durations)
                            self.data[symbol] |= durations
                except websockets.ConnectionClosed:
                    # this reconnects the client with an exponential backoff delay
                    # (because websockets library v10 added async reconnect on continue)
                    logger.error(
                        "[Algo Binder :: {}] Connection dropped, reconnecting...",
                        self.url,
                    )
                    continue
                finally:
                    self.activeWS = None

                logger.error("How did we exit the forever loop?")
        except (KeyboardInterrupt, asyncio.CancelledError, SystemExit):
            # be quiet when manually exiting
            logger.error("[{}] Exiting!", self.url)
            return
        except:
            # on any error, reconnect
            logger.exception("[{}] Can't?", self.url)
            await asyncio.sleep(1)
        finally:
            self.activeWS = None


@functools.lru_cache(maxsize=16)
def convert_time(seconds):
    """Converts the given seconds into a human-readable time format"""

    # Calculate weeks, days, hours, minutes and seconds
    weeks, remainder = divmod(seconds, 604800)
    days, remainder = divmod(remainder, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Create a list to store the formatted time units
    time_units = []

    # Check if each unit is greater than zero and add it to the list if so
    if weeks > 0:
        time_units.append(f"{weeks:.0f} week{'s' if weeks > 1 else ''}")

    if days > 0:
        time_units.append(f"{days:.0f} day{'s' if days > 1 else ''}")

    if hours > 0:
        time_units.append(f"{hours:.0f} hour{'s' if hours > 1 else ''}")

    if minutes > 0:
        time_units.append(f"{minutes:.0f} minute{'s' if minutes > 1 else ''}")

    if seconds > 0 or not time_units:
        time_units.append(f"{seconds:.2f} second{'s' if seconds != 1 else ''}")

    return " ".join(time_units)


def as_duration(seconds):
    """Converts the given seconds into a human-readable time format

    (more compressed format limited to 'days' versus convert_time())
    """

    # Calculate weeks, days, hours, minutes and seconds
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Create a list to store the formatted time units
    time_units = []

    if days > 0:
        time_units.append(f"{days:.0f} d")

    if hours > 0:
        time_units.append(f"{hours:.0f} hr")

    if minutes > 0:
        time_units.append(f"{minutes:.0f} min")

    if seconds > 0 or not time_units:
        time_units.append(f"{seconds:.2f} s")

    return " ".join(time_units)


def analyze_trend_strength(df, ema_col="ema", periods=None):
    """
    Analyzes the directional strength of a time series using multiple timeframes.

    Parameters:
    df: DataFrame with time series data
    ema_col: name of the EMA column to analyze
    periods: list of periods to compare (if None, uses all available rows)

    Returns:
    dict containing trend analysis and strength metrics
    """
    if periods is None:
        # Use all available periods except the last row (current)
        periods = df.index[1].tolist()

    # Calculate changes from current value
    current_value = df[ema_col].iloc[0]
    changes = {
        period: {
            "change": current_value - df[ema_col].loc[period],
            "pct_change": (
                (current_value - df[ema_col].loc[period]) / df[ema_col].loc[period]
            )
            * 100,
        }
        for period in periods
        if not pd.isna(df[ema_col].loc[period])
    }

    # Calculate trend metrics
    changes_array = np.array([v["change"] for v in changes.values()])

    # Overall trend strength metrics
    trend_metrics = {
        "direction": "UP" if np.mean(changes_array) > 0 else "DOWN",
        "strength": abs(np.mean(changes_array)),
        "consistency": np.mean(
            np.sign(changes_array) == np.sign(np.mean(changes_array))
        )
        * 100,
        # for some reason, mypy is reporting np.polyfit doesn't exist when it clearly does
        "acceleration": np.polyfit(range(len(changes_array)), changes_array, 1)[0],  # type: ignore
    }

    # Determine trend phase
    recent_direction = math.copysign(1, changes_array[:3].mean())  # Nearest 3 periods
    overall_direction = math.copysign(1, changes_array.mean())

    if recent_direction > 0 and overall_direction > 0:
        trend_phase = "STRONGLY UP"
    elif recent_direction < 0 and overall_direction < 0:
        trend_phase = "STRONGLY DOWN"
    elif recent_direction > 0 and overall_direction < 0:
        trend_phase = "TURNING UP"
    elif recent_direction < 0 and overall_direction > 0:
        trend_phase = "TURNING DOWN"
    else:
        trend_phase = "NEUTRAL"

    # Calculate trend components
    trend_components = {
        "short_term": math.copysign(1, changes_array[:3].mean()),  # Nearest 3 periods
        "medium_term": math.copysign(
            1, changes_array[: len(changes_array) // 2].mean()
        ),  # First half
        "long_term": math.copysign(1, changes_array.mean()),  # All periods
    }

    return {
        "trend_phase": trend_phase,
        "metrics": trend_metrics,
        "components": trend_components,
        "changes": changes,
    }


def generate_trend_summary(df, ema_col="ema"):
    """
    Generates a human-readable summary of the trend analysis.
    """
    analysis = analyze_trend_strength(df, ema_col)

    strength_desc = (
        "strong"
        if analysis["metrics"]["strength"] > 1
        else "moderate"
        if analysis["metrics"]["strength"] > 0.5
        else "weak"
    )
    consistency_desc = (
        "consistent"
        if analysis["metrics"]["consistency"] > 80
        else "moderately consistent"
        if analysis["metrics"]["consistency"] > 60
        else "inconsistent"
    )

    acceleration_desc = (
        "accelerating"
        if analysis["metrics"]["acceleration"] > 0.1
        else "decelerating"
        if analysis["metrics"]["acceleration"] < -0.1
        else "steady"
    )

    summary = f"The trend is currently in a {analysis['trend_phase']} phase with {strength_desc} momentum. "
    summary += f"The movement is {consistency_desc} across timeframes and is {acceleration_desc}. "

    # Add component analysis
    components = []
    if analysis["components"]["short_term"] > 0:
        components.append("short-term upward")
    if analysis["components"]["medium_term"] > 0:
        components.append("medium-term upward")
    if analysis["components"]["long_term"] > 0:
        components.append("long-term upward")
    if analysis["components"]["short_term"] < 0:
        components.append("short-term downward")
    if analysis["components"]["medium_term"] < 0:
        components.append("medium-term downward")
    if analysis["components"]["long_term"] < 0:
        components.append("long-term downward")

    summary += f"The trend shows {', '.join(components)} movement."

    return summary


def rmsnorm(x, epsilon=1e-6):
    """
    Implements Root Mean Square (RMS) Normalization

    Args:
        x: Input array/tensor to normalize
        epsilon: Small constant for numerical stability (default: 1e-6)

    Returns:
        Normalized array/tensor
    """
    # Calculate the root mean square
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True))

    # Normalize the input
    normalized = x / (rms + epsilon)

    return normalized
