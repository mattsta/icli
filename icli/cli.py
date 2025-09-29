#!/usr/bin/env python3

original_print = print
import asyncio
import bisect
import copy
import datetime
import fnmatch  # for glob string matching!
import itertools
import locale  # automatic money formatting
import logging
import math
import os
import pathlib
import re
import statistics
import sys
import shutil
import time
import pytz

from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import Any, Mapping, Sequence

import bs4
import warnings

# http://www.grantjenks.com/docs/diskcache/
import diskcache  # type: ignore

import numpy as np
import pandas as pd

from pandas.tseries.offsets import MonthEnd, YearEnd, Week
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, ThreadedHistory
from prompt_toolkit.shortcuts import set_title
from prompt_toolkit.styles import Style

import icli.awwdio as awwdio

import icli.calc
import icli.orders as orders

import whenever

from . import instrumentdb
from . import utils

locale.setlocale(locale.LC_ALL, "")

import ib_async

import seaborn  # type: ignore
from ib_async import (
    Bag,
    ComboLeg,
    Contract,
    Future,
    IB,
    IBDefaults,
    Index,
    NewsBulletin,
    NewsTick,
    Order,
    OrderStateNumeric,
    Position,
    PnLSingle,
    RealTimeBarList,
    Ticker,
)

from loguru import logger

import icli.lang as lang
from icli.helpers import *  # FUT_EXP and isset() is appearing from here
import prettyprinter as pp  # type: ignore
import tradeapis.buylang as buylang
import tradeapis.orderlang as orderlang
import tradeapis.ifthen as ifthen
import tradeapis.ifthen_templates as ifthen_templates
import tradeapis.cal as mcal
from tradeapis.ordermgr import OrderMgr, Trade as OrderMgrTrade

from cachetools import cached, TTLCache

from mutil.timer import Timer
from mutil.bgtask import BGTasks, BGSchedule, BGTask

USEastern: Final = pytz.timezone("US/Eastern")

# increase calendar cache duration since we provide exact inputs each time,
# so we know the cache doesn't need to self-invalidate to update new values.
mcal.CALENDAR_CACHE_SECONDS = 60 * 60 * 60

warnings.filterwarnings("ignore", category=bs4.MarkupResemblesLocatorWarning)
pp.install_extras(["dataclasses"], warn_on_error=False)

# environment 1 true; 0 false; flag for determining if EVERY QUOTE (4 Hz per symbol) is saved to a file
# for later backtest usage or debugging (note: this uses the default python 'json' module which sometimes
# outputs non-JSON compliant NaN values, so you may need to filter those out if read back using a different
# json parser)
ICLI_DUMP_QUOTES = bool(int(os.getenv("ICLI_DUMP_QUOTES", 0)))


# setup color gradients we use to show gain/loss of daily quotes
COLOR_COUNT = 100

# palette 'RdYlGn' is a spectrum from low RED to high GREEN which matches
# the colors we want for low/negative (red) to high/positive (green)
MONEY_COLORS = seaborn.color_palette("RdYlGn", n_colors=COLOR_COUNT, desat=1).as_hex()

# only keep lowest 25 and highest 25 elements since middle values are less distinct
MONEY_COLORS = MONEY_COLORS[:25] + MONEY_COLORS[-25:]

# display order we want: RTY / RUT, ES / SPX, NQ / COMP, YM, Index ETFs
FUT_ORD = dict(
    MES=-9,
    ES=-9,
    SPY=-6,
    SPX=-9,
    NANOS=-9,
    RTY=-10,
    RUT=-10,
    M2K=-10,
    IWM=-6,
    NDX=-8,
    COMP=-8,
    NQ=-8,
    QQQ=-6,
    MNQ=-8,
    MYM=-7,
    YM=-7,
    DJI=-7,
    DIA=-6,
)

# A-Z, Z-A, translate between them (lowercase only)
ATOZ = "".join([chr(x) for x in range(ord("a"), ord("z") + 1)])
ZTOA = ATOZ[::-1]
ATOZTOA_TABLE = str.maketrans(ATOZ, ZTOA)


def mkcolor(
    n: float, vals: str | list[str], colorRanges: Sequence[float]
) -> str | list[str]:
    def colorRange(x):
        buckets = len(MONEY_COLORS) // len(colorRanges)
        for idx, crLow in enumerate(colorRanges):
            if x <= crLow:
                return MONEY_COLORS[idx * buckets]

        # else, on the high end of the range, so use highest color
        return MONEY_COLORS[-1]

    # no style if no value (or if nan%)
    if n == 0 or n != n:
        return vals

    # override for high values
    if n >= 0.98:
        useColor = "ansibrightblue"
    else:
        useColor = colorRange(n)

    if isinstance(vals, list):
        return [f"<aaa bg='{useColor}'>{v}</aaa>" for v in vals]

    # else, single thing we can print
    return f"<aaa bg='{useColor}'>{vals}</aaa>"


def mkPctColor(a, b):
    # fmt: off
    colorRanges = [-0.98, -0.61, -0.33, -0.13, 0, 0.13, 0.33, 0.61, 0.98]
    # fmt: on
    return mkcolor(a, b, colorRanges)


@cached(cache={}, key=lambda x, _y: x.conId)
def sortLeg(leg, conIdCache):
    try:
        return (
            conIdCache[leg.conId].right,
            leg.action,
            -conIdCache[leg.conId].strike
            if conIdCache[leg.conId].right == "P"
            else conIdCache[leg.conId].strike,
        )
    except:
        # if cache is broken, just ignore the sort instead of crashing
        return ("Z", leg.action, 0)


# Note: only cache based on the 'x' argument and not the 'contractCache' argument.
@cached(cache={}, key=lambda x, _y: hash(x))
def sortQuotes(x, contractCache: dict[int, Contract] | None = None):
    """Comparison function to sort quotes by specific types we want grouped together."""
    sym, quote = x
    c = quote.contract

    # We want to sort futures first, and sort MES, MNQ, etc first.
    # (also Indexes and Index ETFs first too)
    # This double symbol check is so we don't accidentially sort market ETF options
    # inside the regular equity section.
    if c.secType in {"FUT", "IND", "CONTFUT"} or (
        (c.symbol == c.localSymbol)
        and (
            c.symbol
            in {
                "SPY",
                "UPRO",
                "SPXL",
                "SOXL",
                "SOXS",
                "QQQ",
                "TQQQ",
                "SQQQ",
                "IWM",
                "DIA",
            }
        )
    ):
        priority = FUT_ORD[c.symbol] if c.symbol in FUT_ORD else 0
        return (0, priority, c.secType, c.symbol, c.localSymbol)

    # draw crypto and forex/cash quotes under futures quotes
    if c.secType in {"CRYPTO", "CASH"}:
        priority = 0
        return (0, priority, c.secType, c.symbol, c.localSymbol)

    if c.secType == "OPT":
        # options are medium last because they are wide
        priority = 0
        return (2, priority, c.secType, c.localSymbol, c.symbol)

    if c.secType in {"FOP", "EC"}:
        # future options (and "Event Contracts") are above other options...
        priority = -1

        # Future Options have local symbols like "E4AQ4 C5700" where the full date isn't
        # embedded (just a "Date code" which isn't "sequential in time"), so let's prepend
        # the actual date for sorting these against each other...
        return (
            2,
            priority,
            c.secType,
            c.lastTradeDateOrContractMonth + c.localSymbol,
            c.symbol,
        )

    if c.secType == "BAG":
        # bags are last because their descriptions take up multiple rows
        priority = 0

        # look up PROPERTIES of a bag so we can sort by actual details better...
        if contractCache:
            bagParts = []
            for x in c.comboLegs:
                leg = contractCache.get(x.conId)

                if not leg:
                    break

                bagParts.append(
                    (
                        leg.symbol,
                        leg.lastTradeDateOrContractMonth,
                        x.action[0],
                        leg.right,
                        leg.strike,
                    )
                )

            bagParts = list(sorted(bagParts))

            # logger.info("Bag key: {}", bagKey)
            return (3, priority, c.secType, bagParts, c.symbol)

        return (
            3,
            priority,
            c.secType,
            ":".join(sorted([f"{x.action}-{x.conId}" for x in c.comboLegs])),
            "",
        )

    # else, just by name.
    # BUT we do these in REVERSE order since they
    # are at the end of the table!
    # (We create "reverse order" by translating all
    #  letters into their "inverse" where a == z, b == y, etc).
    priority = 0
    return (1, priority, c.secType, invertstr((c.localSymbol or c.symbol).lower()))


def invertstr(x):
    return x.translate(ATOZTOA_TABLE)


# allow these values to be cached for 10 hours
@cached(cache=TTLCache(maxsize=200, ttl=60 * 90))
def marketCalendar(start, stop):
    return mcal.getMarketCalendar(
        "NASDAQ",
        start=start,
        stop=stop,
    )


# allow these values to be cached for 10 hours
@cached(cache=TTLCache(maxsize=300, ttl=60 * 60 * 10))
def fetchDateTimeOfEndOfMarketDayAtDate(y, m, d):
    """Return the market (start, end) timestamps for the next two market end times."""
    start = pd.Timestamp(y, m, d, tz="US/Eastern")  # type: ignore
    found = marketCalendar(start, start + pd.Timedelta(7, "D"))

    # format returned is two columns of [MARKET OPEN, MARKET CLOSE] timestamps per date.
    soonestStart = found.iat[0, 0]
    soonestEnd = found.iat[0, 1]

    nextStart = found.iat[1, 0]
    nextEnd = found.iat[1, 1]

    return [(soonestStart, soonestEnd), (nextStart, nextEnd)]


def goodCalendarDate():
    """Return the start calendar date we should use for market date lookups.

    Basically, use TODAY if the current time is before liquid hours market close, else use TOMORROW."""
    now = pd.Timestamp("now", tz="US/Eastern")

    # if EARLIER than 4pm, use today.
    if now.hour < 16:
        now = now.floor("D")
    else:
        # else, use tomorrow
        now = now.ceil("D")

    return now


@cached(cache=TTLCache(maxsize=1, ttl=60 * 90))
def tradingDaysRemainingInMonth():
    """Return how many trading days until the month ends...

    NOTE: we are excluding partial/early-close days from the full count..."""
    now = goodCalendarDate()
    found = marketCalendar(now, now + MonthEnd(0))

    found["duration"] = found.market_close - found.market_open
    regularDays = found[found.duration >= pd.Timedelta(hours=6)]

    # just length because the 'found' calendar has one row for each market day in the result set...
    distance = len(regularDays)

    # return ONE LESS THAN trading days found because on the LAST DAY of the month, we want
    # to say there are "0 days" remaining in month, not "1 day remaining" on the last day.
    return distance - 1


@cached(cache=TTLCache(maxsize=1, ttl=60 * 90))
def tradingDaysRemainingInYear():
    """Return how many trading days until the year ends..."""
    now = goodCalendarDate()
    found = marketCalendar(now, now + YearEnd(0))

    distance = len(found)

    # same "minus 1" reason as the days in month
    # (i.e. for the last day in the year, report "0 days remaining in year" not "1 days remaining")
    return distance - 1


@cached(cache=TTLCache(maxsize=20, ttl=60 * 90))
def tradingDaysNextN(days: int):
    """Return calendar dates for the next N trading days"""
    now = goodCalendarDate()

    periods = pd.date_range(now, periods=days)
    found = marketCalendar(periods[0], periods[-1])

    return list(found["market_open"])


# expire this cache once every 15 minutes so we only have up to 15 minutes of wrong dates after EOD
@cached(cache=TTLCache(maxsize=128, ttl=60 * 15))
def fetchEndOfMarketDayAtDate(y, m, d):
    """Return the timestamp of the next end-of-day market timestamp.

    This is currently only used for showing the "end of day" countdown timer in the toolbar,
    so it's okay if we return an expired date for a little while (the 15 minute cache interval),
    so the toolbar will just report a negative closing time for up to 15 minutes.

    The cache structure is because the toolbar refresh code is called anywhere from 1 to 10 times
    _per second_ so we want to minimize as much math and logic overhead as possible for non-changing
    values.

    We could potentially place an event timer somewhere to manually clear the cache at EOD,
    but we just aren't doing it yet."""
    [(soonestStart, soonestEnd), (nextStart, nextEnd)] = (
        fetchDateTimeOfEndOfMarketDayAtDate(y, m, d)
    )

    # this logic just helps us across the "next day" barrier when this runs right after a normal 4pm close
    # so we immediately start ticking down until the next market day close (which could be 3-4 days away depending on holidays!)
    if soonestEnd.timestamp() > whenever.ZonedDateTime.now("US/Eastern").timestamp():
        return whenever.ZonedDateTime.from_timestamp(
            soonestEnd.timestamp(), tz="US/Eastern"
        )

    return whenever.ZonedDateTime.from_timestamp(nextEnd.timestamp(), tz="US/Eastern")


# Fields updated live for toolbar printing.
# Printed in the order of this list (the order the dict is created)
# Some math and definitions for values:
# https://www.interactivebrokers.com/en/software/tws/usersguidebook/realtimeactivitymonitoring/available_for_trading.htm
# https://ibkr.info/node/1445
LIVE_ACCOUNT_STATUS: Final = [
    # row 1
    "AvailableFunds",
    # NOTE: we replaced "BuyingPower" with a 3-way breakdown instead:
    "BuyingPower4",
    "BuyingPower3",
    "BuyingPower2",
    "Cushion",
    "DailyPnL",
    "DayTradesRemaining",
    "DayTradesRemainingT+1",
    "DayTradesRemainingT+2",
    "DayTradesRemainingT+3",
    "DayTradesRemainingT+4",
    # row 2
    "ExcessLiquidity",
    "FuturesPNL",
    "GrossPositionValue",
    "MaintMarginReq",
    "OptionMarketValue",
    "EquityWithLoanValue",
    # row 3
    "NetLiquidation",
    "RealizedPnL",
    "TotalCashValue",
    "UnrealizedPnL",
    "SMA",
    # unpopulated:
    #    "Leverage",
    #    "HighestSeverity",
]

# we need to add extra keys for VERIFICATION, but we don't show these extra keys directly in the status bar...
STATUS_FIELDS_PROCESS: Final = set(LIVE_ACCOUNT_STATUS) | {"BuyingPower"}


def readableHTML(html) -> str:
    """Return contents of 'html' with tags stripped and in a _reasonably_
    readable plain text format.

    This is used for printing "IBKR Realtime Status Updates/News" from the API.
    The API sends news updates as HTML, so we convert it to text for terminal display.
    """

    return re.sub(
        r"(\n[\s]*)+", "\n", bs4.BeautifulSoup(html, features="html.parser").get_text()
    )


stocks = ["QQQ", "SPY", "AAPL"]

# Futures to exchange mappings:
# https://www.interactivebrokers.com/en/index.php?f=26662
# Note: Use ES and RTY and YM for quotes because higher volume
#       also curiously, MNQ has more volume than NQ?
# Volumes at: https://www.cmegroup.com/trading/equity-index/us-index.html
# ES :: MES
# RTY :: M2K
# YM :: MYM
# NQ :: MNQ
sfutures = {
    "CME": ["ES", "RTY", "MNQ", "GBP"],  # "HE"],
    "CBOT": ["YM"],  # , "TN", "ZF"],
    #    "NYMEX": ["GC", "QM"],
}

# Discovered via mainly: https://www.linnsoft.com/support/symbol-guide-ib
# The DJI / DOW / INDU quotes don't work.
# The NDX / COMP quotes require differen't data not included in default packages.
#    Index("COMP", "NASDAQ"),
idxs = [
    Index("SPX", "CBOE"),
    # No NANOS because most brokers don't offer it and it has basically no volume
    # Index("NANOS", "CBOE"),  # SPY-priced index options with no multiplier
    Index("VIN", "CBOE"),  # VIX Front-Month Component (near term)
    Index("VIF", "CBOE"),  # VIX Front-er-Month Component (far term)
    Index("VIX", "CBOE"),  # VIX Currently (a mix of VIN and VIF basically)
    # No VOL-NYSE because it displays billions of shares and breaks our views
    # Index("VOL-NYSE", "NYSE"),
    Index("TICK-NYSE", "NYSE"),
    # > 1 == selling pressure, < 1 == buying pressure; somewhat
    Index("TRIN-NYSE", "NYSE"),
    # Advancing minus Declining (bid is Advance, ask is Decline) (no idea what the bid/ask qtys represent)
    Index("AD-NYSE", "NYSE"),
]

# Note: ContFuture is only for historical data; it can't quote or trade.
# So, all trades must use a manual contract month (quarterly)
# TODO: we should be consuming a better expiration date system because some
#       futures expire end-of-month (interest rates), others quarterly (indexes), etc.
futures = [
    Future(
        symbol=sym,
        lastTradeDateOrContractMonth=FUT_EXP or "",
        exchange=x,
        currency="USD",
    )
    for x, syms in sfutures.items()
    for sym in syms
]


@dataclass(slots=True)
class IBKRCmdlineApp:
    # Your IBKR Account ID (required)
    accountId: str

    # number of seconds between refreshing the toolbar quote/balance views
    # (more frequent updates requires higher CPU utilization for the faster redrawing)
    toolbarUpdateInterval: float = 2.22

    host: str = "127.0.0.1"
    port: int = 4001

    # global client ID for your IBKR gateway connection (must be unique per client per gateway)
    clientId: int = field(default_factory=lambda: int(os.getenv("ICLI_CLIENT_ID", 0)))
    customName: str = field(default_factory=lambda: os.getenv("ICLI_CLIENT_NAME", ""))

    # initialized to True/False when we first see the account
    # ID returned from the API which will tell us if this is a
    # sandbox ID or True Account ID
    isSandbox: bool | None = None

    # The Connection
    ib: IB = field(
        default_factory=lambda: IB(
            defaults=IBDefaults(
                emptyPrice=None, emptySize=None, unset=None, timezone=USEastern
            )
        )
    )

    # count total toolbar refreshes
    updates: int = 0

    # same as 'updates' except this resets to 0 if your session gets disconnected then reconnected
    updatesReconnect: int = 0

    now: whenever.ZonedDateTime = field(
        default_factory=lambda: whenever.ZonedDateTime.now("US/Eastern")
    )
    nowpy: datetime.datetime = field(default_factory=lambda: datetime.datetime.now())

    quotesPositional: list[tuple[str, ITicker]] = field(default_factory=list)
    dispatch: lang.Dispatch = field(default_factory=lang.Dispatch)

    # holder for background events being run for some purpose
    tasks: BGTasks = field(init=False)

    # our own order tracking!
    ordermgr: OrderMgr = field(init=False)

    # Timed Events!
    scheduler: BGTasks = field(init=False)

    # use a single calculator instance so we only need to parse the grammar once
    calc: icli.calc.Calculator = field(init=False)

    # generic cache for data usage (strikes, etc)
    cache: Mapping[Any, Any] = field(
        default_factory=lambda: diskcache.Cache("./cache-multipurpose")
    )

    # global state variables (set per-client and per-session currently with no persistence)
    # We also populate the defaults here. We can potentially have these load from a config
    # file instead of being directly stored here.
    localvars: dict[str, str] = field(default_factory=lambda: dict(exchange="SMART"))

    # State caches
    quoteState: dict[str, ITicker] = field(default_factory=dict)
    contractIdsToQuoteKeysMappings: dict[int, str] = field(default_factory=dict)
    depthState: dict[Contract, Ticker] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)
    pnlSingle: dict[int, PnLSingle] = field(default_factory=dict)
    exiting: bool = False

    # cache some parsers. yes these names are confusing. sorry.
    ol: buylang.OLang = field(default_factory=buylang.OLang)
    requestlang: orderlang.OrderLang = field(default_factory=orderlang.OrderLang)

    # global ifthenRuntime for all data processing and predicate execution
    ifthenRuntime: ifthen.IfThenRuntime = field(default_factory=ifthen.IfThenRuntime)

    # maps of tempalte names to template executor instances. We have one executor per "template type"
    # we then sub-populate with more concrete symbol/algo details so we can run one template multiple
    # times with different arugments (i.e. multiple symbols trading under the same tempalte logic, etc)
    ifthenTemplates: ifthen_templates.IfThenMultiTemplateManager = field(init=False)

    # cache recent IBKR API event messages so they don't overwhelm the console
    # (some IBKR order warning/error messages repeat aggressively for multiple minutes even though they don't
    #  reall matter, so we report them once every N occurrences or time period instead of printing thousands
    #  of lines of exact duplicate warnings all at once)
    duplicateMessageHandler: utils.DuplicateMessageHandler = field(
        default_factory=utils.DuplicateMessageHandler
    )

    # in-progress: attempt in-process paper trading space for fake order tracking.
    paperLog: dict[str, PaperLog] = field(default_factory=lambda: defaultdict(PaperLog))

    # track our own custom position representations...
    # (NOTE: one 'IPosition' _may_ belong to multiple 'ContractId' values.
    #        For spreads, we assign each contract id leg to the SAME IPosition representating the spread.
    #        This also means you will get weird/broken behavior if you have contracts internal and
    #        external to spreads).
    # Also, we currently only use 'IPosition' objects for _active_ accumulation/distribution sessions,
    # so these values do not need to persist across restarts.
    iposition: dict[Contract, IPosition] = field(default_factory=dict)

    # something any interested party can await on a contract to detect when it has new COMPLETED orders.
    # Note: this only fires when an order has qtyRemaining==0 (so it fires on each ORDER, not each _execution_).
    fillers: dict[Contract, CompleteTradeNotification] = field(
        default_factory=lambda: defaultdict(CompleteTradeNotification)
    )

    # our API-validated price increment database for every instrument so we can determine
    # proper limit order price tick increments before submitting orders.
    idb: instrumentdb.IInstrumentDatabase = field(init=False)

    # Say hello to our 3rd attempt at consuming live externally-generated algo datafeeds into our trading process...
    algobinder: AlgoBinder | None = None
    algobindertask: BGTask | None = None

    speak: awwdio.AwwdioClient = field(default_factory=awwdio.AwwdioClient)

    # Specific dict of ONLY fields we show in the live account status toolbar.
    # Saves us from sorting/filtering self.summary() with every full bar update.
    accountStatus: dict[str, float] = field(
        default_factory=lambda: dict(
            zip(LIVE_ACCOUNT_STATUS, [0.00] * len(LIVE_ACCOUNT_STATUS))
        )
    )

    # Cache all contractIds and names to their fully qualified contract object values
    # TODO: replace this with our dual in-memory-disk-passthrough cache.
    conIdCache: diskcache.Cache = field(
        default_factory=lambda: diskcache.Cache("./cache-contracts")
    )

    connected: bool = False
    disableClientQuoteSnapshotting: bool = False
    loadingCommissions: bool = False

    toolbarStyle: Style = field(
        default_factory=lambda: Style.from_dict(
            {"bottom-toolbar": "fg:default bg:default"}
        )
    )

    opstate: Any = field(init=False)

    def algobinderStart(self) -> bool:
        """Returns True if we started the algobinder.
        Returns False if algobinder was already running.
        """
        if not self.algobinder:
            self.algobinder = AlgoBinder()

        if not self.algobindertask:
            logger.info("Starting algo binder task...")
            self.algobindertask = self.task_create(
                "Algo Binder Data Receiver", self.algobinder.datafeed()
            )

            return True

        return False

    def algobinderStop(self) -> bool:
        """Returns True if we stopped the algobinder live processing task (data remains available, just not updating anymore).
        Returns False if there is no active algobinder to stop.
        """

        if self.algobindertask:
            logger.info("Stopping algo binder task...")
            self.algobindertask.stop()
            self.algobindertask = None
            return True

        return False

    def updateToolbarStyle(self, val: str) -> None:
        """Create new style object when style text"""

        assert isinstance(val, str)

        # note: add 'noreverse' to make bg=bg and fg=fg, otherwise it treats bg=fg and fg=bg
        # bg #33363D is nice, but needs a lighter font
        # kinda nice (camo green-ish) #708C4C
        # https://ethanschoonover.com/solarized/#the-values
        # Solarized(ish): {"bottom-toolbar": "fg:#002B36 bg:#839496"}

        # Want to add your own custom theme? Submit an issue with good color combinations and we'll add it!
        schemes: Final = dict(
            default="fg:default bg:default", solar1="fg:#002B36 bg:#839496"
        )

        # if input is a theme name, use the theme colors
        if theme := schemes.get(val):
            logger.info("Setting toolbar style ({}): {}", val, theme)
            val = theme
        else:
            logger.info("Setting toolbar style: {}", theme)

        self.toolbarStyle = Style.from_dict({"bottom-toolbar": val})

    def tradingDays(self, days):
        return tradingDaysNextN(days)

    @property
    def diy(self) -> int:
        """Return remaining trading days in year"""
        return tradingDaysRemainingInYear()

    @property
    def dim(self) -> int:
        """Return remaining trading days in month"""
        return tradingDaysRemainingInMonth()

    def __post_init__(self) -> None:
        # just use the entire IBKRCmdlineApp as our app state!
        self.opstate = self
        self.setupLogging()

        # attach runtime to multi-template manager (since we can't attach the runtime at field() init time)
        self.ifthenTemplates = ifthen_templates.IfThenMultiTemplateManager(
            self.ifthenRuntime
        )

        # note: ordermgr is NOT scoped per-client because all clients can see all positions.
        self.ordermgr = OrderMgr(f"Executed Positions")

        self.tasks = BGTasks(f"icli client {self.clientId} internal")
        self.scheduler = BGTasks(f"icli client {self.clientId} scheduler")

        # provide ourself to the calculator so the calculator can lookup live quote prices and live account values
        self.calc = icli.calc.Calculator(self)

        # provide ourself to instrumentdb so it can also use live API calls
        self.idb = instrumentdb.IInstrumentDatabase(self)

    def setupLogging(self) -> None:
        # Configure logger where the ib_insync live service logs get written.
        # Note: if you have weird problems you don't think are being exposed
        # in the CLI, check this log file for what ib_insync is actually doing.
        now = pd.Timestamp("now")
        LOGDIR = (
            pathlib.Path(os.getenv("ICLI_LOGDIR", "runlogs"))
            / f"{now.year}"
            / f"{now.month:02}"
        )
        LOGDIR.mkdir(exist_ok=True, parents=True)
        LOG_FILE_TEMPLATE = str(
            LOGDIR
            / f"icli-id={self.clientId}-{whenever.ZonedDateTime.now('US/Eastern').py_datetime()}".replace(
                " ", "_"
            )
        )
        logging.basicConfig(
            level=logging.INFO,
            filename=LOG_FILE_TEMPLATE + "-ibkr.log",
            format="%(asctime)s %(message)s",
        )

        logger.info("Logging session with prefix: {}", LOG_FILE_TEMPLATE)

        def asink(x):
            # don't use print_formatted_text() (aliased to print()) because it doesn't
            # respect the patch_stdout() context manager we've wrapped this entire
            # runtime around. If we don't have patch_stdout() guarantees, the interface
            # rips apart with prompt and bottom_toolbar problems during async logging.
            original_print(x, end="")

        logger.remove()
        logger.add(asink, colorize=True)

        # new log level to disable color bolding on INFO default
        logger.level("FRAME", no=25)
        logger.level("ARGS", no=40, color="<blue>")

        # Also configure loguru logger to log all activity to its own log file for historical lookback.
        # also, these are TRACE because we log _user input_ to the TRACE facility, but we don't print
        # it to the console (since the user already typed it in the console)
        logger.add(sink=LOG_FILE_TEMPLATE + "-icli.log", level="TRACE", colorize=False)
        logger.add(
            sink=LOG_FILE_TEMPLATE + "-icli-color.log",
            level="TRACE",
            colorize=True,
        )

    async def qualify(self, *contracts, overwrite: bool = False) -> list[Contract]:
        """Qualify contracts against the IBKR allowed symbols.

        Mainly populates .localSymbol and .conId

        You can set 'overwrite' if you want to ALWAYS recache these lookups.

        We also cache the results for ease of re-use and for mapping
        contractIds back to names later."""

        # logger.info("Inbound request (overwrite: {}): {}", overwrite, contracts)
        # logger.warning("Current full cache: {}", [x for x in self.conIdCache])

        # Group contracts into cached and uncached so we can look up uncached contracts
        # all at once but still iterate them in expected order of results.
        cached_contracts = {}
        uncached_contracts = []

        # in order to retain the result in the same order as the input, we map the python id() value of
        # each contract object to the final contract result itself. Then, at the end, we just iterate
        # the input 'contracts' collection looking up them by id() for the final result order matching
        # the input order again.
        totalResult = {}

        def cachedContractCorrupt(cached_contract, contract) -> bool:
            if cached_contract and (not cached_contract.conId):
                logger.warning(
                    "BUG: Why doesn't cached contract have an ID? Looking up again. Request: {} vs Found: {}",
                    contract,
                    cached_contract,
                )

                return True

            # watch out for data bugs if we had a mis-placed cache element somehow
            if cached_contract and (
                (contract.strike and (contract.strike != cached_contract.strike))
                or (contract.secType and (contract.secType != cached_contract.secType))
                or (
                    contract.localSymbol
                    and (contract.localSymbol != cached_contract.localSymbol)
                )
            ):
                logger.warning(
                    "BUG: Cached contract doesn't match requested contract? Rejecting cache and looking up again. Request: {} vs. Found: {}",
                    contract,
                    cached_contract,
                )

                return True

            if type(cached_contract) == Contract:
                logger.warning(
                    "BUG: Cached contract is just 'Contract' which doesn't work. Rejecting cache and looking up again. Request: {} vs Found: {}",
                    contract,
                    cached_contract,
                )

                return True

            return False

        # Our cache operates on two tiers:
        #  - ideally, we look up contracts by id directly
        #  - alternatively, we look up contracts by name, but we also need to know the _type_ of the name we are looking up.
        # So, we check the requested contracts for:
        #  - if input contract already has a contract id, we look up the conId directly.
        #  - if input contract doesn't have an id, we generate a lookup key of Class-Symbol like "Future-ES" or "Index-SPX"
        #    so we can retrieve the correct instrument class combined with the proper symbol from a cached contract.
        for contract in contracts:
            cached_contract = None

            assert isinstance(
                contract, Contract
            ), f"Why didn't you send a contract here? Note: the input is *contracts, and _not_ a list of contracts! Got: {contract}"

            key = None
            try:
                # only attempt to look up using ID if ID exists, else attempt to lookup by name
                if contract.conId:
                    # key = contractToSymbolDescriptor(contract)
                    # logger.info("by ID Looking up: {} :: {}", contract, key)
                    # Attempt first lookup using direct ID, but if ID isn't found try to use the Class-Symbol key format...
                    cached_contract = self.conIdCache.get(contract.conId)  # type: ignore
                else:
                    key = contractToSymbolDescriptor(contract)
                    # logger.info("by KEY Looking up: {} :: {}", contract, key)
                    cached_contract = self.conIdCache.get(key)  # type: ignore

                # logger.info("Using cached contract for {}: [cached {}] :: [requested {}]", contract.conId, cached_contract, contract)
            except ModuleNotFoundError:
                # the pickled contract is from another library we don't have loaded in this environment anymore,
                # so we need to drop the existing bad pickle and re-save it
                try:
                    del self.conIdCache[contract.conId]
                    del self.conIdCache[contractToSymbolDescriptor(contract)]
                except:
                    pass

            # if a manual refresh, never use a cached contract to force a new lookup
            if overwrite:
                cached_contract = None

            if cachedContractCorrupt(cached_contract, contract):
                cached_contract = None

            # if we _found_ a contract (and the contract has an id (just defensively preventing invalid contracts in the cache)),
            # then we don't look it up again.
            if cached_contract and cached_contract.conId:
                # logger.info("Found in cache: {} for {}", cached_contract, contract)
                cached_contracts[cached_contract.conId] = cached_contract
                totalResult[id(contract)] = cached_contract
            else:
                # else, we need to look up this contract before returning.
                # logger.info("Not found in cache: {}", contract)
                # Also, save the ORIGINAL LOOK UP KEY along with the uncached contract so we can
                # _correctly_ map the lookup key to the resolved contract (the resolved contract
                # can (and _will_ have more details than the lookup contract, but we only want to
                # generate the lookup key using the INPUT DETAILS and not the FULL DETAILS because
                # we are going from VAGUE DETAILS -> SPECIFIC DETAILS, and if we use the specific
                # details as the cache key, we can't look it up again because we only have more
                # vague details to start (like: lookup future with YM expiration, but qualify
                # converts YM into YMD, but we didn't look up YMD at first, so we must not cache
                # by using YMD details in the key, etc).

                # always populate unresolved contract for safety in case it can't be resolved
                # we just return it directly as originally provided
                totalResult[id(contract)] = contract

                # also, don't look up Bag contracts because they don't qualify (the legs *inside* the bag qualify instead)
                if not isinstance(contract, Bag):
                    uncached_contracts.append((key, contract))

        # logger.info("CACHED: {} :: UNCACHED: {}", cached_contracts, uncached_contracts)

        # if we have NO UNCACHED CONTRACTS, then we found all input requests in the cache,
        # so we can just return everything we already recorded as a "result contract" (_including_ unqualified bags).
        if not uncached_contracts:
            return [totalResult[id(c)] for c in contracts]

        # For uncached, fetch them from the IBKR lookup system
        got = []
        try:
            # logger.info("Looking up uncached contracts: {}", uncached_contracts)

            # iterate requests in smaller blocks if we have a large input request
            CHUNK = 50

            # Note: "Bag" contracts can NEVER be qualified, so don't ever try them (avoid a timeout wait if bags are attempted)
            for block in range(0, len(uncached_contracts), CHUNK):
                # logger.info("CHECKING: {}", pp.pformat(uncached_contracts))
                logger.info(
                    "Qualifying {} contracts from {} to {}...",
                    len(uncached_contracts),
                    block,
                    block + CHUNK,
                )
                got.extend(
                    await asyncio.wait_for(
                        self.ib.qualifyContractsAsync(
                            *[
                                c
                                for (k, c) in uncached_contracts[block : block + CHUNK]
                            ],
                            returnAll=True,
                        ),
                        timeout=min(
                            6, 2 * len(uncached_contracts[block : block + CHUNK])
                        ),
                    )
                )

            # logger.info("Got: {}", got)
        except Exception as e:
            logger.error(
                "Timeout while trying to qualify {} contracts (sometimes IBKR is slow or the API is offline during nightly restarts) :: {}",
                len(uncached_contracts),
                str(e),
            )

        assert (
            len(got) == len(uncached_contracts)
        ), f"We can't continue caching if we didn't lookup all the contracts! {len(got)=} vs {len(uncached_contracts)=}"

        # iterate resolved contracts and cache them by multiple lookup keys
        for (originalContractKey, requestContract), contract in zip(
            uncached_contracts, got
        ):
            # don't cache continuous futures contracts because those are only for quotes and not trading
            # if contract.secType == "CONTFUT":
            #    continue

            if isinstance(contract, list):
                logger.error(
                    "[{}] contract request returned multiple matches! Can't determine which one of {} to cache: {}",
                    requestContract,
                    len(contract),
                    pp.pformat(contract),
                )
                continue

            # if this lookup was a failure, don't attempt to cache anything...
            if not contract:
                logger.error("No contract ID resolved for: {}", requestContract)
                continue

            if not contract.conId:
                logger.error("No contract ID resolved for: {}", contract)
                continue

            # sometimes we end up with a fully populated Contract() we want to make more specific,
            # so _only_ run this check if type is Contract and _not_ a sub-class of Contract
            # (which is why we must use `type(c) is Contract` and not isinstance(c, Contract))
            if type(contract) is Contract:
                contract = Contract.recreate(contract)

            # final verification check the fields match as expected
            if cachedContractCorrupt(contract, requestContract):
                logger.error("Failed to qualify _actual_ Contract type?")
                continue

            # the `qualifyContractsAsync` modifies the contracts in-place, so we map their
            # id to itself since we replaced it directly.
            # (yes, we _always_ set this even if we didn't resolve a 'conId' because we need
            #  to return _all_ contracts back to the user in the order of their inputs, so
            #  we need every input contract to be in the 'totalResult' map regardless of its final
            #  success/fail resolution value)
            totalResult[id(contract)] = contract

            if type(contract) == Contract:
                # Convert generic 'Contract' to its actual underlying type for proper storage and future retrieval
                contract = Contract.create(**ib_async.util.dataclassAsDict(contract))

                # update key too
                originalContractKey = contractToSymbolDescriptor(contract)

            # Note: this is correct because we want to check for EXACT contract matches and not any subclass of Contract
            if type(contract) == Contract:
                logger.warning(
                    "Not caching because Contract isn't a specific type: {}", contract
                )
                continue

            # Only cache actually qualified contracts with a full IBKR contract ID
            if not contract.conId:
                continue

            # We added double layers of sanity checking here because we had some cache data anomalies where
            # the incorrect contract was cached into the wrong id. These should detect and prevent it from happening
            # again if our logic changes didn't catch all the edge cases.
            if requestContract.strike:
                if requestContract.strike != contract.strike:
                    logger.error(
                        "Why didn't resolved contract have the same strike as the input contract? [request {}] vs [qualified {}]",
                        requestContract,
                        contract,
                    )

                    continue

            cached_contracts[contract.conId] = contract

            # we want Futures contracts to refresh more often because they have
            # embedded expiration dates which may change over time if we are using
            # generic symbol names like "ES" for the next main contract.
            EXPIRATION_DAYS = 5 if isinstance(contract, Future) else 90

            # logger.info("Saving {} -> {}", contract.conId, contract)
            # logger.info("Saving {} -> {}", originalContractKey, contract)

            if False:
                logger.info("Setting {} -> {}", contract.conId, contract)
                logger.info("Setting {} -> {}", originalContractKey, contract)
                logger.info(
                    "Setting {} -> {}",
                    (contract.localSymbol, contract.symbol),
                    contract,
                )

            # cache by id
            assert contract.conId

            # TODO: make the expiration time more clever where it picks an expiration time at 5pm eastern after hours.
            self.conIdCache.set(
                contract.conId, contract, expire=86400 * EXPIRATION_DAYS
            )  # type: ignore

            # also set by Class-Symbol designation as key (e.g. "Index-SPX" or "Future-ES")
            self.conIdCache.set(
                originalContractKey,
                contract,
                expire=86400 * EXPIRATION_DAYS,
            )  # type: ignore

            # also cache the same thing by the most well defined symbol we have
            self.conIdCache.set(
                (contract.localSymbol, contract.symbol),
                contract,
                expire=86400 * EXPIRATION_DAYS,
            )  # type: ignore

        # Return in the same order as the input by combining cached and uncached results.
        # NOTE: we DO NOT MODIFY THE CACHED CONTRACT RESULTS IN-PLACE so you must assign the
        #       return value of this async call to be your new list of contracts.
        result = [totalResult[id(c)] for c in contracts]

        assert len(result) == len(
            contracts
        ), "Why is result length different than request length?"

        # logger.info("Returning contracts: {}", result)
        return result

    def updateGlobalStateVariable(self, key: str, val: str | None) -> None:
        # 'val' of None means just print the output, while 'val' of empty string means delete the key.
        if val is None:
            logger.info("No value provided, so printing current settings:")
            for k, v in sorted(self.localvars.items()):
                logger.info("SET: {} = {}", k, v)

            return

        original = self.localvars.get(key)

        if val:
            # if value provided, set it

            # special values if setting dte things
            if key.lower() == "dte":
                now = pd.Timestamp("now")

                if not (val.isnumeric()):
                    # pandas weekdays are indexed by Monday == 0
                    match val.lower():
                        case "monday" | "mon" | "m":
                            weekday = 0
                        case "tuesday" | "tues" | "t":
                            weekday = 1
                        case "wednesday" | "wed" | "w":
                            weekday = 2
                        case "thursday" | "thurs" | "th":
                            weekday = 3
                        case "friday" | "fri" | "f":
                            weekday = 4
                        case _:
                            logger.error("DTE values are weekdays only!")
                            return

                    # calcluate number of calendar days between now and the requested expiration day.
                    # NOTE: Due to how we automatically make "after 4pm == 0dte", this 'dte-by-day-of-week'
                    #       doesn't work correctly after hours because it will always be one extra day ahead.
                    #       (e.g. Monday after hours, our 0dte is tuesday, but calendar tuesday is 1 day away, so 1 dte == wednesday)
                    val = ((now + Week(weekday=weekday)) - now).days  # type: ignore

            self.localvars[key] = val
        else:
            # else, if value not provided, remove key
            self.localvars.pop(key, None)

        if original and not val:
            logger.info("UNSET: {} (previously: {})", key, original)
        elif original:
            logger.info("SET: {} = {} (previously: {})", key, val, original)
        else:
            logger.info("SET: {} = {}", key, val)

    def contractsForPosition(
        self, sym, qty: float | None = None
    ) -> list[tuple[Contract, float, float]]:
        """Returns matching portfolio positions as list of (contract, size, marketPrice).

        Note: input 'sym' can be a glob pattern for symbol matching. '?' matches single character, '*' matches any characters.

        Looks up position by symbol name (allowing globs) and returns either provided quantity or total quantity.
        If no input quantity, return total position size.
        If input quantity larger than position size, returned size is capped to max position size.
        """
        portitems = self.ib.portfolio()
        # logger.debug("Current Portfolio is: {}", portitems)

        results = []
        for pi in portitems:
            # Note: using 'localSymbol' because for options, it includes
            # the full OCC-like format, while contract.symbol will just
            # be the underlying equity symbol.
            # Note note: using fnmatch.fnmatch() because we allow 'sym' to
            #            have glob characters for multiple lookups at once!
            # Note 3: options .localSymbols have the space padding, so remove for input compare.
            # TODO: fix temporary hack of OUR symbols being like /NQ but position values dont' have the slash...
            if fnmatch.fnmatch(
                pi.contract.localSymbol.replace(" ", ""), sym.replace("/", "")
            ):
                contract = None
                contract = pi.contract
                position = pi.position

                if qty is None:
                    # if no quantity requested, use entire position
                    foundqty = position
                elif abs(qty) >= abs(position):
                    # else, if qty is larger than position, truncate to position.
                    foundqty = position
                else:
                    # else, use requested quantity but with sign of position
                    foundqty = math.copysign(qty, position)

                # note: '.marketPrice' here is IBKR's "best effort" market price because it only
                #       updates maybe every 30-90 seconds? So (qty * .marketPrice * multiplier) may not represent the
                #       actual live value of the position.
                results.append((contract, foundqty, pi.marketPrice))

        return results

    async def contractForOrderRequest(
        self, oreq: buylang.OrderRequest
    ) -> Contract | None:
        """Return a valid qualified contract for any order request.

        If order request has multiple legs, returns a Bag contract representing the spread.
        If order request only has one symbol, returns a regular future/stock/option contract.

        If symbol(s) in order request are not valid, returns None."""

        if oreq.isSpread():
            return await self.bagForSpread(oreq)

        if oreq.isSingle():
            contract = contractForName(oreq.orders[0].symbol)
            # logger.info("Contracting: {}", contract)

            if contract:
                (contract,) = await self.qualify(contract)

                # only return success if the contract validated
                if contract.conId:
                    return contract

            return None

        # else, order request had no orders...
        return None

    async def bagForSpread(self, oreq: buylang.OrderRequest) -> Contract | Bag | None:
        """Given a multi-leg OrderRequest, return a qualified Bag contract.

        If legs do not validate, returns None and prints errors along the way."""

        # For IBKR spreads ("Bag" contracts), each leg of the spread is qualified
        # then placed in the final contract instead of the normal approach of qualifying
        # the final contract itself (because Bag contracts have Legs and each Leg is only
        # a contractId we have to look up via qualify() individually).
        contracts = [
            contractForName(
                s.symbol,
            )
            for s in oreq.orders
        ]
        contracts = await self.qualify(*contracts)

        # if the bag only has one contract, just return it directly and avoid actually creating a bag.
        if len(contracts) == 1:
            return contracts[0]

        if not all([c.conId for c in contracts]):
            logger.error("Not all contracts qualified! Got: {}", contracts)
            return None

        # trying to match logic described at https://interactivebrokers.github.io/tws-api/spread_contracts.html
        underlyings = ",".join(sorted({x.symbol for x in contracts}))

        # Iterate (in MATCHED PAIRS) the resolved contracts with their original order details
        legs = []

        # We want to order the purchase legs as:
        #   - Option or FuturesOption first (protection first)
        #   - anything else later (underlying later)
        # So, if security type is OPT or FOP, use a smaller value so they SORT FIRST (just negative contract integers so it's always lower!)
        # then regular contract ids for anything else...
        useExchange: str
        for c, o in sorted(
            zip(contracts, oreq.orders),
            key=lambda x: -x[0].conId if x[0].secType in {"OPT", "FOP"} else x[0].conId,
        ):
            useExchange = c.exchange
            leg = ComboLeg(
                conId=c.conId,
                ratio=o.multiplier,
                action="BUY" if o.isBuy() else "SELL",
                exchange=c.exchange,
            )

            legs.append(leg)

        bag = Bag(
            symbol=underlyings,
            comboLegs=legs,
            currency="USD",
        )

        # use SMART if mixing security types.
        bag.exchange = useExchange if (await self.isGuaranteedSpread(bag)) else "SMART"

        return bag

    def quoteResolve(self, lookup: str) -> tuple[str, Contract] | tuple[None, None]:
        """Resolve a local symbol alias like ':33' to current symbol name for the index."""

        # TODO: this doesn't work for futures symbols. Probably need to read the contract type
        #       to re-apply or internal formatting? futs: /; CFD: CFD; crypto: C; ...
        # TODO: fix this lookup if the number doesn't exist. (e.g. deleting :40 when quote 40 isn't valid
        #       results in looking up ":"[1:] which is just empty and it breaks.
        #       Question though: what do we return when a quote doesn't exist? Does the code using this method accept None as a reply?

        # extract out the number only here... (_ASSUMING_ we were called correct with ':33' etc and not just '33')
        lookupId = lookup[1:]

        if not lookupId:
            return None, None

        try:
            lookupInt = int(lookupId)
            _lookupsymbol, ticker = self.quotesPositional[lookupInt]
        except:
            # either the input wasn't ':number' or the index doesn't exist...
            return None, None

        # now we passed the integer extraction and the quote lookup, so return the found symbol for the lookup id
        assert ticker.contract
        name = (ticker.contract.localSymbol or ticker.contract.symbol).replace(" ", "")

        return name, ticker.contract

    def decimals(self, contract: Contract) -> int:
        """How many decimal places should a contract use?"""
        # if no decimal specification found, default to 2.
        if (digits := self.idb.decimals(contract)) is None:
            return 2

        # always use 2 digits for display even if only 1 digit of precision is required
        if digits == 1:
            return 2

        return digits

    async def tickIncrement(self, contract: Contract) -> Decimal | None:
        """Dynamically calculate the tick increment for 'contract' by assuming we want the lowest price above zero it can be."""
        return await self.complyUp(contract, Decimal("0.00001"))

    async def comply(
        self, contract: Contract, price: Decimal | float, direction: instrumentdb.ROUND
    ) -> Decimal | None:
        """Given a contract and an estimated price, round the price to a value appropriate for the instrument."""
        return await self.idb.round(contract, price, direction)

    async def complyNear(
        self, contract: Contract, price: Decimal | float
    ) -> Decimal | None:
        """Given a contract and an estimated price, round price to NEAREST value appropriate for the instrument."""
        return await self.idb.round(contract, price, instrumentdb.ROUND.NEAR)

    async def complyUp(
        self, contract: Contract, price: Decimal | float
    ) -> Decimal | None:
        """Given a contract and an estimated price, round price to equal OR HIGHER value appropriate for the instrument."""
        return await self.idb.round(contract, price, instrumentdb.ROUND.UP)

    async def complyDown(
        self, contract: Contract, price: Decimal | float
    ) -> Decimal | None:
        """Given a contract and an estimated price, round price to equal OR LOWER value appropriate for the instrument."""
        return await self.idb.round(contract, price, instrumentdb.ROUND.DOWN)

    async def safeModify(self, contract, order, **kwargs) -> Order:
        """Given a current order, generate a new order we can _safely_ use to submit for modification.

        This is needed because we can't just re-use an existing trade.order object. The IBKR API dynamically
        back-populates live metadata into the cached trade order object, and we can't send all those auto-populated
        fields back as order modification updates.

        So here we have a centralized way of "cleaning up" an existing trade record to generate a new order which won't
        generate API errors when submitted (hopefully)."""

        # fmt: off
        updatedOrder = dataclasses.replace(
            order,

            # IBKR rejects updates if we submit "IBALGO" as the order type (but IBKR also back-populates the order type to 'IBALGO' itself, so we have to _remove_ it every time)
            orderType="LMT" if order.orderType == "IBALGO" else order.orderType,

            # now add user-requested updates
            **kwargs
        )
        # fmt: on

        # IBKR populates the 'volatility' field on some orders, but it also rejects order updates
        # if the 'volatility' field has a value when the order type is not a VOLATILITY order itself.
        # This covers orders named: VOL and these others below.
        # There are also other VOL orders, but it's unclear if they use the volatility parameter or not:
        # PEGMIDVOL,PEGMKTVOL,PEGPRMVOL,PEGSRFVOL,VOLAT
        if "VOL" not in updatedOrder.orderType:
            updatedOrder.volatility = None

        # also, order updates cannot have parentIds even if they were originally submitted with them (as far as we can tell based on error messages)
        if updatedOrder.parentId:
            updatedOrder.parentId = 0

        # Cached original orders from brackets sometimes have the executing order as Transmit=False, but once the order is live, Transmit must be True always.
        updatedOrder.transmit = True

        # IBKR auto-popualtes fields in live orders we want to remove in future updates if we aren't actively providing them
        removeIfNotRequested = (
            "adjustedOrderType",
            "clearingIntent",
            "deltaNeutralOrderType",
            "displaySize",
            "dontUseAutoPriceForHedge",
            "trailStopPrice",
            "volatilityType",
        )

        for key in removeIfNotRequested:
            if key not in kwargs:
                setattr(updatedOrder, key, None)

        if isset(updatedOrder.lmtPrice):
            fixprice = await self.comply(
                contract,
                updatedOrder.lmtPrice,
                instrumentdb.ROUND.UP
                if updatedOrder.action == "BUY"
                else instrumentdb.ROUND.DOWN,
            )

            if fixprice != updatedOrder.lmtPrice:
                logger.warning(
                    "Updated limit price to comply with order mintick from {} to {}",
                    updatedOrder.lmtPrice,
                    fixprice,
                )
                updatedOrder.lmtPrice = fixprice

        if isset(updatedOrder.auxPrice):
            fixprice = await self.comply(
                contract,
                updatedOrder.auxPrice,
                instrumentdb.ROUND.UP
                if updatedOrder.action == "BUY"
                else instrumentdb.ROUND.DOWN,
            )

            if fixprice != updatedOrder.auxPrice:
                logger.warning(
                    "Updated aux price to comply with order mintick from {} to {}",
                    updatedOrder.auxPrice,
                    fixprice,
                )
                updatedOrder.auxPrice = fixprice

        return updatedOrder

    async def fetchContractExpirations(
        self, contract: Contract, fetchDates: list[str] | None = None
    ):
        """Abstract a dual-use API preferring to use Tradier for strike fetching because IBKR data processing is awful.

        fetchDates is optional if using tradier fetching because tradier returns all expiration dates for a symbol, but
        IBKR requires per-date fetching per contract (and providing wider dates like YYYYMM returns an entire month of
        chains, but also invokes IBKR data pacing limitations because their data formats are big and they are slow)."""

        # only run for regular Options on Stock-like things (SPX counts as "Stock" here for our lookups too)
        if os.getenv("TRADIER_KEY") and isinstance(contract, Stock):
            if found := await getExpirationsFromTradier(contract.symbol):
                return found

        strikes: dict[str, list[float]] = defaultdict(list)
        allStrikes: dict[str, list[float]] = dict()

        assert fetchDates
        fetchDates = sorted(set(fetchDates))
        logger.info("[{}] Requested dates: {}", contract, fetchDates)

        if False:
            # TODO: use this as a fallback if the regualr lookups don't work.... mainly for indexes?
            # pre-lookup
            everything = await asyncio.wait_for(
                self.ib.reqSecDefOptParamsAsync(
                    contract.symbol, "CME", contract.secType, contract.conId
                ),
                timeout=10,
            )

            logger.info(
                "Also found: {}",
                pp.pformat(everything),
            )

            exchanges = sorted(set([x.exchange for x in everything]))
            logger.info("Valid exchanges: {}", pp.pformat(exchanges))

            if contract.secType == "IND":
                if exchanges:
                    contract.exchange = exchanges[0]

        for date in fetchDates:
            contract.lastTradeDateOrContractMonth = date
            chainsExact = await asyncio.wait_for(
                self.ib.reqContractDetailsAsync(contract), timeout=180
            )

            # group strike results by date
            logger.info(
                "[{}{}] Populating strikes...",
                contract.localSymbol,
                contract.lastTradeDateOrContractMonth,
            )

            for d in sorted(
                chainsExact,
                key=lambda k: k.contract.lastTradeDateOrContractMonth,  # type: ignore
            ):
                assert d.contract
                strikes[d.contract.lastTradeDateOrContractMonth].append(
                    d.contract.strike
                )

            # cleanup the results because they were received in an
            # arbitrary order, but we want them sorted for bisecting
            # and just nice viewing.
            allStrikes |= {k: sorted(set(v)) for k, v in strikes.items()}

        return allStrikes

    async def isGuaranteedSpread(self, bag: Contract) -> bool:
        # Note: only STK+OPT or OPT+OPT spreads are guaranteed. Other instrument bags are but not executed atomically and may partially execute.
        # Also note: if using 'SMART' due to conflicting instrument types, the Order for execution must be marked NonGuaranteed.

        # We need to fetch the contracts from the contract ids since this is just a bag with ids and we don't know contract types... thanks API.
        legs = await self.qualify(*[Contract(conId=x.conId) for x in bag.comboLegs])

        secTypes = set([x.secType for x in legs])

        # Single instrument routing is safe
        if len(secTypes) == 0:
            return True

        # a spread is guaranteed only for bags with stock and option combinations.
        # So, if we have more than one security type, but everything is STK and OPT, we are still Guaranteed.
        if len(secTypes) > 1 and len({"STK", "OPT"} - secTypes) > 0:
            return False

        # else, we just checked all legs are either STK or OPT, so we are guaranteed
        return True

    async def contractForOrderSide(self, order: Order, contract: Contract) -> Contract:
        """If required, return a copy of 'contract' with legs in the direction of the order.

        This is only reuqired for non-guaranteed spreads where we need to execute protection before acquiring or selling underlyings.
        """

        if isinstance(contract, Bag):
            # for BUYING, we need the protective options FIRST
            # for SELLING, we need the protective options SECOND

            # return a COPY of the contract here because we need different contracts for different order sides.
            contract = copy.copy(contract)

            def optionScoreFirst(leg, secType):
                """If security type is an option, use a lower sort key, else use a higher sort key for non-options"""
                return -leg.conId if secType in {"OPT", "FOP"} else leg.conId

            async def sortLegsDirection(legs, optionsFirstMultiplier=1):
                # option legs are only contract ids, so we need to look up the actual contracts to figure out what each leg actually is
                cs = await self.qualify(*[Contract(conId=x.conId) for x in legs])

                # we need to sort the legs, but legs don't have details, so we combine the legs with the contracts, sort the pairs,
                # then we un-combine them and return only the legs (in correct order now)
                resorted = sorted(
                    zip(cs, legs),
                    # this basically just keeps options first if multiper == 1 or options last if multiplier == -1
                    key=lambda x: optionsFirstMultiplier
                    * optionScoreFirst(x[1], x[0].secType),
                )

                return list(zip(*resorted))[1]

            async def sortLegsOptionFirst(legs):
                # we need to look up ids to contracts to figure out what they are... thanks IBKR API
                return await sortLegsDirection(legs, 1)

            async def sortLegsOptionLast(legs):
                return await sortLegsDirection(legs, -1)

            match order.action:
                case "BUY":
                    # sort protection FIRST
                    contract.comboLegs = await sortLegsOptionFirst(contract.comboLegs)
                case "SELL":
                    # sort protection LAST
                    contract.comboLegs = await sortLegsOptionLast(contract.comboLegs)

        return contract

    async def addNonGuaranteeTagsIfRequired(self, contract, *reqorders):
        if isinstance(contract, Bag):
            # only check contract once up front if we need to actually use the 'not guaranteed' fields
            # (non-guaranteed means IBKR may execute the spread sequentially instead of as a single price-locked unit)
            isGuaranteed = await self.isGuaranteedSpread(contract)

            # if contract is guaranteed (default behavior), it executes as expected so we don't need to modify the orders
            if isGuaranteed:
                return

            # else, we need to add custom "non-guaranteed" config flags/tags to each order
            for order in reqorders:
                # only send orders if the order is populated (we get _all_ potential limit/profit/loss orders here, but the bracket may not be defined)
                if order:
                    orders.markOrderNotGuaranteed(order)

    def createBracketAttachParent(
        self,
        order,
        sideClose,
        qty,
        profitLimit,
        lossLimit,
        lossStopPrice,
        outsideRth,
        tif,
        orderTypeProfit,
        orderTypeLoss,
        config=None,
    ) -> tuple[Order | None, Order | None]:
        """Given a starting order, generate bracket take profit/stop loss orders attached to the starting order.

        Return value is (profitOrder, lossOrder) but either or both orders may be None if parameters are not defined.

        Yes, these parameters are a bit of a mess, but it works for now since we want to use this logic in more than one place.
        Technically this could consume a full OrderIntent object and just do the correct thing itself along with a parent order a time params.

        Note: only run this if you have AT LEAST ONE profit or loss order to attach.
        """
        # When creating attached orders, we need manual order IDs because by default they only
        # get generated during the order placement phase.
        if not order.orderId:
            order.orderId = self.ib.client.getReqId()

        order.transmit = False

        profitOrder = None
        lossOrder = None
        if profitLimit is not None:
            profitOrder = orders.IOrder(
                sideClose,
                qty,
                profitLimit,
                outsiderth=outsideRth,
                tif=tif,
                config=config,
            ).order(orderTypeProfit)
            assert profitOrder

            profitOrder.orderId = self.ib.client.getReqId()
            profitOrder.parentId = order.orderId
            profitOrder.transmit = False

        if lossLimit is not None:
            assert lossStopPrice
            lossOrder = orders.IOrder(
                sideClose,
                qty,
                lossLimit,
                aux=lossStopPrice,
                outsiderth=outsideRth,
                tif=tif,
                config=config,
            ).order(orderTypeLoss)
            assert lossOrder

            lossOrder.orderId = self.ib.client.getReqId()
            lossOrder.parentId = order.orderId
            lossOrder.transmit = False

        # if loss order exists, it ALWAYS transmits last
        if lossOrder:
            lossOrder.transmit = True
        elif profitOrder:
            # else, only PROFIT ORDER exists, so send it (profit order is ignored)
            profitOrder.transmit = True

        return profitOrder, lossOrder

    async def placeOrderForContract(
        self,
        sym: str,
        isLong: bool,
        contract: Contract,
        qty: PriceOrQuantity,
        limit: Decimal | None,
        orderType: str,
        preview: bool = False,
        # This may be getting out of hand, but for the priority is:
        #  - If 'bracket' and 'ladder' exists, it's an error.
        #  - Always use 'config' for orders.
        #  - If 'ladder' exists, generate a chain of parent orders for each step
        #    ending in a final average price stop loss.
        bracket: Bracket | None = None,
        config: Mapping[str, Decimal | float] | None = None,
        ladder: Ladder | None = None,
    ) -> FullOrderPlacementRecord | None:
        """Place a BUY (isLong) or SELL (not isLong) for qualified 'contract' at 'qty' for 'limit' (if applicable).

        The 'qty' parameter allows switching between price amounts and share/contract/quantity amounts directly.
        """

        # if bracket *AND* ladder are provided, we _remove_ the bracket because
        # the bracket is _included_ in the ladder, but we _do not_ generate a bracket
        # for the initial order itself here, the bracket is _only_ for the average
        # price to stop out or take profit on the ladder if it gets that far.
        if bool(bracket) and bool(ladder):
            bracket = None

        # if we have an exchange override, use it here. If not, the contract exchange is not altered.
        if qty.exchange:
            # Note: if this contract has NOT had a detail request run against it, we can get a false exchange listing the first request,
            #       (like, "SMART ONLY" even though it has a dozen other exchanges), but it _will_ populate in the background for a future
            #       run to likely work properly again.
            xs = self.idb.exchanges(contract)

            # TODO: this exchange validator may need to be a top-level helper method for exchange confirmation in other places too.
            if qty.exchange not in xs:
                logger.error(
                    "[{} :: {}] Requested exchange not found for instrument! Requested exchange: {}\nAvailable exchanges: {}",
                    contract.secType,
                    sym,
                    qty.exchange,
                    pp.pformat(sorted(xs)),
                )

                return None

            contract.exchange = qty.exchange

        logger.info(
            "[{} :: {}] Using exchange: {}",
            contract.symbol,
            contract.localSymbol,
            contract.exchange,
        )

        # Immediately ask to add quote to live quotes for this trade positioning...
        # turn option contract lookup into non-spaced version
        sym = contract.localSymbol.replace(" ", "") or sym.replace(" ", "")

        if limit is not None:
            if isLong:
                comply = await self.complyUp(contract, limit)
            else:
                comply = await self.complyDown(contract, limit)

            assert comply is not None
            limit = comply

        if qty.is_quantity and not limit:
            logger.info("[{}] Request to order qty {} at current prices", sym, qty)
        else:
            digits = self.decimals(contract) or 2
            logger.info(
                "[{}] Request to order at dynamic qty {} @ price {:,.{}f}",
                sym,
                qty,
                limit,
                digits,
            )

        quotesym = sym

        # if quote isn't live, add it so we can check against bid/ask details
        self.addQuoteFromContract(contract)

        if not contract.conId:
            # spead contracts don't have IDs, so only reject if NOT a spread here.
            if not isinstance(contract, Bag):
                logger.error(
                    "[{} :: {}] Not submitting order because contract not qualified!",
                    sym,
                    quotesym,
                )
                return None

        if isinstance(contract, Bag):
            # steal multiplier of first thing in contract. we assume it's okay? This would be wrong for buy-write bags and is only valid for spreads.
            (innerContract,) = await self.qualify(
                Contract(conId=contract.comboLegs[0].conId)
            )
        else:
            innerContract = contract

        multiplier = self.multiplier(contract)

        # REL and LMT/MKT/MOO/MOC orders can be outside RTH, but futures trade without RTH designation all the time
        # Futures have no "RTH" so they always execute if markets are open.
        if isinstance(innerContract, (FuturesOption, Future)):
            outsideRth = False
        else:
            outsideRth = True

        if isinstance(contract, Option):
            # Purpose: don't trigger warning about "RTH option has no effect" with options...
            if contract.localSymbol[0:3] not in {"SPX", "VIX"}:
                # Currently only SPX and VIX options trade outside (extended) RTH, but other things don't,
                # so turn the flag off so the IBKR Order system doesn't generate a warning
                # considered "outside RTH."
                # For SPY, QQQ, IWM, SMH, and other ETFs, RTH is considered to end at 1615.
                outsideRth = False

        # Note: don't make this an 'else if' to the previous check because this needs to also run again
        # for all option types.
        # TODO: make this list more exhaustive for what only works during RTH liquid hours. Often the IBKR
        #       order system will just say "well, this flag is wrong and I'm ignoring it to execute the order anyway..."
        if orderType in {
            "MIDPRICE",
            "MKT + ADAPTIVE + FAST",
            "MKT + ADAPTIVE + SLOW",
            "LMT + ADAPTIVE + FAST",
            "LMT + ADAPTIVE + SLOW",
        }:
            # as a usability helper, if we are trying to AF or AS on futures, just LMT instead because
            # IBKR rejects attempts to use adaptive algo orders on CME exchanges apparently.
            if isinstance(innerContract, FuturesOption):
                orderType = "LMT"
            else:
                # TODO: cleanup, also verify how we want to run FAST or EVICT outside RTH?
                # Algos can only operate RTH:
                outsideRth = False

                logger.warning(
                    "[{}] ALGO NOT SUPPORTED FOR ALL HOURS. ORDER RESTRICTED TO RTH ONLY!",
                    orderType,
                )

        tif: Literal["Minutes", "DAY", "GTC"]
        if isinstance(contract, Crypto) and isLong:
            # Crypto can only use IOC or Minutes for tif BUY
            # (but for SELL, can use IOC, Minutes, Day, GTC)
            tif = "Minutes"
        elif contract.exchange in {"OVERNIGHT", "IBEOS"}:
            # Overnight requests can't persist past the 20:00-03:50 session (vampire orders!)
            tif = "DAY"
        else:
            # TODO: Add default TIF capability also to global var setting? Or let it be configured in the "limit" menu too?
            tif = "GTC"

        determinedQty: Decimal | float | int = 0

        # if input is quantity, use quantity directly
        # TODO: also allow quantity trades to submit their own limit price like 100@3.33???
        # Maybe even "100@3.33+" to start with _our_ limit price, but then run our price-follow-tracking algo
        # if the initial offer doesn't execute after a couple seconds?
        if qty.is_quantity:
            determinedQty = qty.qty

        bid: None | float = None
        ask: None | float = None
        # Also, this loop does quote lookup to generate the 'limit' price if none exists.
        # Conditions:
        #  - if quantity is a dollar amount, we need to calculate quantity based on current quote.
        #  - also, if this is a preview (with or without a limit price), we calculate a price for margin calculations.
        #  - basically: guard against quantity orders attempting to lookup prices when they aren't needed.
        #    (market orders also imply quantity is NOT money because a market order with no quantity doesn't make sense)
        if ((not limit) and ("MKT" not in orderType)) or preview:
            # only use our automatic-midpoint if we don't already have a limit price
            quoteKey = lang.lookupKey(contract)

            # if this is a new quote just requested, we may need to wait
            # for the system to populate it...
            loopFor = 10

            # only show this quote loop if: LIVE REQUEST or REQUESTING DYNAMIC LIMIT PRICE
            while not (
                currentQuote := self.currentQuote(
                    quoteKey, show=(not (preview or limit))
                )
            ):
                logger.warning(
                    "[{} :: {}] Waiting for quote to populate...", quoteKey, loopFor
                )
                try:
                    await asyncio.sleep(0.033)
                except:
                    logger.warning("Cancelled waiting for quote...")
                    return None

                if (loopFor := loopFor - 1) == 0:
                    # if we exhausted the loop, we didn't get a usable quote so we can't
                    # do the requested price-based position sizing.
                    logger.error("[{}] No live quote available?", quoteKey)

                    # if we have a limit price, use it as the synthetic quote if a quote isn't available
                    if limit is not None:
                        currentQuote = (float(limit), float(limit))
                        break

                    # no price and no quote, so we can't do anything else here
                    return None

            assert currentQuote
            bid, ask = currentQuote

            if bid is None and ask is None:
                # just make mypy happy with float/Decimal potential differences
                assert limit is not None
                baL = float(limit)
                bid, ask = baL, baL
            elif limit is None:
                assert ask is not None

                if bid is None:
                    logger.warning(
                        "[{}] WARNING: No bid price, so just using ASK directly for buying!",
                        quoteKey,
                    )
                    bid = ask

                # TODO: need customizable aggressiveness levels
                #   - midpoint (default)
                #   - ask + X% for aggressive time sensitive buys
                #   - bid - X% for aggressive time sensitive sells
                # TODO: need to create active management system to track growing/shrinking
                #       midpoint for buys (+price, -qty) or sell (-price) targeting.
                #       See: lang: "buy" for price tracking after order logic.

                # calculate current midpoint of spread rounded to 2 decimals.
                # FAKE THE MIDPOINT WITH A BETTER MARKET BUFFER
                # If we do *exact* midpoint and prices are rapidly rising or falling, we constantly miss
                # the fills. So give it a generous buffer for quicker filling.
                # (could aso just do MKT or MKT PRT orders too in some circumstances)
                # (LONG means allow HIGHER prices for buying (worse entries the higher it goes);
                #  SHORT means allow LOWER prices for selling (worse entries the lower it goes)).
                # We expect the market NBBO to be our actual bounds here, but we're adjusting the
                # overall price for quicker fills.

                # Note: this logic is different than the direct 'evict' logic where we place wider limit
                #       bounds in an attempt to get out as soon as possible. This is more "at market, best effort,
                #       and follow the price if we don't get it the first time" attempts.
                if isinstance(contract, Option):
                    # Options retain "regular" midpoint behavior because spreads can be wide and hopefully
                    # quotes are fairly slow/stable.
                    mid = (bid + ask) / 2
                else:
                    # equity, futures, etc get the wider margins
                    # NOTE: this looks backwards because for us to ACQUIRE a position we must be BETTER than the market
                    #       on limit prices, so here we have BUY HIGH and SELL LOW just to get the position at first.
                    # TODO: these offsets need to be more adaptable to recent ATR-like conditions per symbol,
                    #       but the goal here is immediate fills at market-adjusted prices anyway.
                    # TODO: compare against automaticLimitBuffer() for setting values here???
                    mid = ((bid + ask) / 2) * (1.005 if isLong else 0.995)

                # we checked 'limit is None' in this branch, so we are safe to set/overwrite limit here.
                limit = await self.complyNear(contract, mid)
                assert limit is not None

        # only update qty if this is a money-ask because we also use this limit discovery
        # for quantity-only orders, where we don't want to alter the quantity, obviously.
        if qty.is_money:
            amt = qty.qty

            # calculate order quantity for spend budget at current estimated price
            logger.info("[{}] Trying to order ${:,.2f} worth at {}...", sym, amt, qty)

            assert limit is not None
            determinedQty = self.quantityForAmount(contract, amt, limit)

            if not determinedQty:
                logger.error(
                    "[{}] Zero quantity calculated for: {} {} {}!",
                    sym,
                    contract,
                    amt,
                    limit,
                )
                return None

            assert determinedQty > 0

            logger.info(
                "Ordering {:,} {} at ${:,.2f} for ${:,.2f}",
                determinedQty,
                sym,
                limit,
                Decimal(determinedQty) * limit * Decimal(multiplier),
            )

        # declare default values so we can check against them later...
        profitOrder = None
        lossOrder = None

        try:
            sideOpen: BuySell = "BUY" if isLong else "SELL"
            sideClose: BuySell = "SELL" if isLong else "BUY"

            # add instrument-specific digits only to price data (not qty or multipler data)
            digits = self.decimals(contract) or 2
            logger.info(
                "[{} :: {}] {:,.2f} @ ${:,.{}f} x {:,.0f} ({}) ALL_HOURS={} TIF={}",
                orderType,
                sideOpen,
                determinedQty,
                limit,
                digits,
                multiplier,
                fmtmoney(float(determinedQty) * float(limit or 0) * multiplier),
                outsideRth,
                tif,
            )

            order = orders.IOrder(
                sideOpen,
                float(determinedQty),
                limit,
                outsiderth=outsideRth,
                tif=tif,
                config=config,
            ).order(orderType)
            assert order

            if bracket:
                profitOrder, lossOrder = self.createBracketAttachParent(
                    order,
                    sideClose,
                    float(determinedQty),
                    bracket.profitLimit,
                    bracket.lossLimit,
                    bracket.lossStop,
                    outsideRth,
                    tif,
                    bracket.orderProfit,
                    bracket.orderLoss,
                )
        except:
            logger.exception("ORDER GENERATION FAILED. CANNOT PLACE ORDER!")
            return None

        # if ladder requested, create a tier of orders each parented to the previous.
        steps: list[tuple[str, Order]] = []
        if ladder:
            prevId = None
            prevOrder = None
            for i, step in enumerate(ladder):
                # generate a new order ID on each order step
                currentId = self.ib.client.getReqId()

                # get current price and quantity of step
                p = step.limit
                q = step.qty
                stepname = f"STEP {i}"

                # Create order for step
                steporder = orders.IOrder(
                    sideOpen,
                    q,
                    p,
                    outsiderth=outsideRth,
                    tif=tif,
                    config=config,
                ).order(orderType)
                assert steporder

                steporder.orderId = currentId
                steporder.transmit = False

                # attach future orders to be children of the previous order
                if prevOrder:
                    steporder.parentId = prevOrder.orderId

                prevOrder = steporder

                steps.append((stepname, steporder))

            profitprice = ladder.profit
            stopprice = ladder.stop

            # if we have at least one of take profit or stop loss, engage the bracket logic.
            if stopprice or profitprice:
                profitprice = (
                    await self.comply(
                        innerContract,
                        profitprice,
                        instrumentdb.ROUND.DOWN if isLong else instrumentdb.ROUND.UP,
                    )
                    if profitprice
                    else None
                )
                stopprice = (
                    await self.comply(
                        innerContract,
                        stopprice,
                        instrumentdb.ROUND.DOWN if isLong else instrumentdb.ROUND.UP,
                    )
                    if stopprice
                    else None
                )

                profitOrder, lossOrder = self.createBracketAttachParent(
                    prevOrder,
                    sideClose,
                    ladder.qty,
                    profitprice,
                    stopprice,
                    stopprice,
                    outsideRth,
                    tif,
                    # TODO: make these also adjutable... we probably want a "STOP LIMIT" here instead of just STP since we have multiple qty
                    ladder.profitAlgo,
                    ladder.stopAlgo,
                    config,
                )

                # Add profit take and loss reducer to orders for placement.
                steps.extend(
                    [
                        (n, o)
                        for n, o in [("PROFIT", profitOrder), ("LOSS", lossOrder)]
                        if o is not None
                    ]
                )
            else:
                steps[-1][-1].transmit = True

            # else, if we do not have ANY profit or loss requested, we must update the final order to transmit=True so the entire chain executes.

        if order.orderType == "PEG MID":
            if isinstance(contract, Option):
                logger.warning(
                    "[{}] Routing order to IBUSOPT for PEG MID",
                    contract.localSymbol or contract.symbol,
                )
                contract.exchange = "IBUSOPT"
            elif isinstance(contract, Stock):
                logger.warning(
                    "[{}] Routing order to IBKRATS for PEG MID",
                    contract.localSymbol or contract.symbol,
                )
                contract.exchange = "IBKRATS"
            else:
                logger.error("Peg-to-Midpoint is only valid for Stocks and Options!")
                return None

        name = contract.localSymbol or contract.symbol
        desc = f"{name} :: QTY {order.totalQuantity:,}"

        # convert quantity to integer (away from float) if it is the same for nicer formatting
        if order.totalQuantity == (itq := int(order.totalQuantity)):
            order.totalQuantity = itq

        ordpairs: tuple[tuple[str, Order | None], ...]
        if preview:
            # generate input format to preview report (tuple of tuples of (name, order))
            if steps:
                # if we have steps created, the 'steps' are already in ordpairs format.
                # (except, we want the FIRST order to be LAST, so we reverse all of these here)
                ordpairs = tuple(reversed(steps))
            else:
                # default profit/loss/order
                ordpairs = (
                    ("PROFIT", profitOrder),
                    ("LOSS", lossOrder),
                    ("TRADE", order),
                )
            runOrders = tuple(
                [
                    (ordname, ordord)
                    for ordname, ordord in ordpairs
                    if ordord is not None
                ]
            )

            # logger.info("Orders are: {}", pp.pformat(runOrders))

            await self.generatePreviewReport(contract, bid, ask, runOrders, multiplier)

            # preview request complete! Nothing remaining to do here.
            return None

        logger.info("[{}] Ordering {} via {}", desc, contract, order)

        profitTrade = None
        lossTrade = None

        # placeOrder() returns a "live updating" Trade object with live position execution detail updates

        if isinstance(contract, Bag):
            await self.addNonGuaranteeTagsIfRequired(
                contract, order, profitOrder, lossOrder
            )

        trade = self.ib.placeOrder(
            await self.contractForOrderSide(order, contract), order
        )

        limitRecord = TradeOrder(trade, order)

        profitTrade = None
        profitRecord = None
        if profitOrder:
            profitTrade = self.ib.placeOrder(
                await self.contractForOrderSide(profitOrder, contract), profitOrder
            )

            profitRecord = TradeOrder(profitTrade, profitOrder)

        lossTrade = None
        lossRecord = None
        if lossOrder:
            lossTrade = self.ib.placeOrder(
                await self.contractForOrderSide(lossOrder, contract), lossOrder
            )

            lossRecord = TradeOrder(lossTrade, lossOrder)

        assert trade
        logger.info(
            "[{} :: {} :: {}] Placed: {}",
            trade.orderStatus.orderId,
            trade.orderStatus.status,
            name,
            pp.pformat(trade),
        )

        if profitOrder:
            assert profitTrade
            logger.info(
                "[{} :: {} :: {}] Profit Order Placed: {}",
                profitTrade.orderStatus.orderId,
                profitTrade.orderStatus.status,
                name,
                pp.pformat(profitTrade),
            )

        if lossOrder:
            assert lossTrade
            logger.info(
                "[{} :: {} :: {}] Loss Order Placed: {}",
                lossTrade.orderStatus.orderId,
                lossTrade.orderStatus.status,
                name,
                pp.pformat(lossTrade),
            )

        # create event for clients to listen for in-progress execution updates until
        # the entire quantity of this order is filled.
        return FullOrderPlacementRecord(
            limitRecord, profit=profitRecord, loss=lossRecord
        )

    async def generatePreviewReport(
        self,
        contract: Contract,
        bid: float | None,
        ask: float | None,
        orders: Sequence[tuple[str, Order]],
        multiplier: float = 1,
    ) -> None:
        # we assume the final order in the list is the ACTUAL INITIAL TRADE ORDER
        # Note: order previews don't respect the OCA or OTOCO or parentId system, so they all just get run independently,
        #       but we use the price of the *last* element in the order list to be the initial live order we are attempting.
        order = orders[-1][-1]
        previewPrice = order.lmtPrice
        assert previewPrice is not None

        # Note: we require space removal here because we use 'no-space-symbols' for quote lookups later.
        symname = (contract.localSymbol or contract.symbol).replace(" ", "")
        desc = f"{symname} :: QTY {order.totalQuantity:,}"

        def whatIfPrepare(ordName, o):
            # preview orders ignore brackets and always transmit by default (the IBKR API won't preview transmit=False orders)
            # Also, the IBKR order preview system doesn't know what to do with parent-tracked since it doesn't maintain an order submission queue.
            logger.info(
                "[{} :: {:>6} :: {}] For preview query, converting: [transmit {}] to False; [parentId {}] to None",
                desc,
                ordName,
                o.orderId,
                o.transmit,
                o.parentId,
            )
            o.transmit = True
            o.parentId = None

            # logger.info("Sending order: {}", pp.pformat(o))
            return o

        if isinstance(contract, Bag):
            await self.addNonGuaranteeTagsIfRequired(contract, *[o[1] for o in orders])

        try:
            # if this is a multi-order bracket, run all the whatIf requests concurrently (a 2x-3x speedup over sequential operations)
            orderStatusResults = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        self.ib.whatIfOrderAsync(contract, whatIfPrepare(name, check))
                        for name, check in orders
                    ]
                ),
                timeout=2,
            )

            assert orderStatusResults
            whatIfResults = tuple(zip(orders, orderStatusResults))
        except:
            logger.error(
                "Timeout while trying to run order preview (sometimes IBKR is slow or the order preview API could be offline)"
            )
            return

        assert whatIfResults

        # If order has negative price to desginate a short, flip it back to negative quanity for our reporting metrics to work.
        # Note: this is safe because preview orders are never processed as 'live' orders, so modifying the orders in-place is okay.
        for name, o in orders:
            if o.lmtPrice is not None and o.lmtPrice < 0:
                o.totalQuantity = -o.totalQuantity
                o.lmtPrice = -o.lmtPrice

        # preview EACH part of the potential 3-way backet order.
        # Note: we process 'TRADE' last, leaving 'status' as the final trade status for the final preview math after this loop.
        for (ordName, order), statusStrs in whatIfResults:  # type: ignore
            if not statusStrs:
                logger.warning(
                    "No preview status generated? Can't process preview request!"
                )
                return

            # request all fields as numeric types instead of default strings (so we can do math on the results easier)
            status: OrderStateNumeric = statusStrs.numeric(digits=2)

            logger.info(
                "[{} :: {}] PREVIEW REQUEST {} via {}",
                desc,
                ordName,
                contract,
                pp.pformat(order),
            )

            logger.info(
                "[{} :: {}] PREVIEW RESULT: {}",
                desc,
                ordName,
                pp.pformat(status.formatted()),
            )

        isContract = isinstance(contract, (Bag, Option, Future, FuturesOption))

        # We currently assume only two kinds of things exist. We could add more.
        nameOfThing = "CONTRACT" if isContract else "SHARE"

        # set 100% margin defaults so our return value has something populated even if margin isn't relevant (options, etc)
        margPctInit = 100.0
        margPctMaint = 100.0

        digits = self.decimals(contract)

        # fix up math issues if totalQuantity became a Decimal() along the way
        order.totalQuantity = float(order.totalQuantity)

        # TODO: we still need to fix ib_async to return None for unset fields of Order() object, but it's more defaults-rewrite work.
        if isset(previewPrice):
            logger.info(
                "[{}] PREVIEW LIMIT PRICE PER {}: ${:,.{}f} (actual @ {}x: ${:,.{}f})",
                desc,
                nameOfThing,
                previewPrice,
                digits,
                multiplier,
                float(previewPrice or 0) * multiplier,
                digits,
            )

        # for options or other conditions, there's no margin change to report.
        # also, if there is a "warning" on the trade, the numbers aren't valid.
        if (
            (not status.warningText)
            and (status.initMarginChange > 0)
            and previewPrice is not None
        ):
            assert order
            baseTotal = order.totalQuantity * float(previewPrice) * multiplier
            margPctInit = (
                status.initMarginChange / (baseTotal or status.initMarginChange)
            ) * 100
            margPctMaint = (
                status.maintMarginChange / (baseTotal or status.maintMarginChange)
            ) * 100

            # if this order is for a CREDIT spread, our margin calculations don't apply because the margin
            # is _reserved_ on our account instead of "paid forward" as with equity symbols.
            # Basically, only show margin requirements if account is debited for this transaction.
            logger.info(
                "[{}] PREVIEW MARGIN REQUIREMENT INIT: {:,.2f} % ({})",
                desc,
                margPctInit,
                fmtmoney(status.initMarginChange),
            )

            # "MAIN" for "MAINTENANCE" to match the length of "INIT" above for alignment.
            # FIX WITH PRT:
            # 2023-11-17 06:55:20.473 | INFO     | icli.cli:placeOrderForContract:834 - [RTYZ3 :: QTY 6] PREVIEW MARGIN REQUIREMENT INIT: 0.00 %
            # 2023-11-17 06:55:20.474 | INFO     | icli.cli:placeOrderForContract:841 - [RTYZ3 :: QTY 6] PREVIEW MARGIN REQUIREMENT MAIN: 0.00 % (IBKR is loaning 100.00 %)
            logger.info(
                "[{}] PREVIEW MARGIN REQUIREMENT MAIN: {:,.2f} % (IBKR is loaning {:,.2f} %)",
                desc,
                margPctMaint,
                100 - margPctMaint,
            )

            if margPctInit and status.initMarginChange >= status.maintMarginChange:
                logger.info(
                    "[{}] PREVIEW MARGIN REQUIREMENT DRAWDOWN ALLOWED: {:,.2f} %",
                    desc,
                    100 * (1 - margPctMaint / margPctInit),
                )

                logger.info(
                    "[{}] PREVIEW MARGIN REQUIREMENT DRAWDOWN LEVERAGE POINTS: {:,.{}f}",
                    desc,
                    (status.initMarginChange - status.maintMarginChange)
                    / multiplier
                    / order.totalQuantity,
                    digits,
                )

                logger.info(
                    "[{}] PREVIEW MAINT MARGIN PER {}: {}",
                    desc,
                    nameOfThing,
                    fmtmoney(status.maintMarginChange / order.totalQuantity),
                )

            logger.info(
                "[{}] PREVIEW INIT MARGIN PER {}: {}",
                desc,
                nameOfThing,
                fmtmoney(status.initMarginChange / order.totalQuantity),
            )

        leverageKind = "CONTRACT" if isContract else "STOCK"
        assert order

        # estimate gains if the obtained quantity moves by specific price increment amounts.
        # Note: only report leverage if order is taking on risk (e.g. don't report for closing transactions).
        if status.equityWithLoanChange <= 0 or status.initMarginChange > 0:
            for amt in (0.20, 0.75, 1, 3, 5):
                logger.info(
                    "[{}] PREVIEW LEVERAGE ({:,} x {}): ${:,.2f} {} MOVE LEVERAGE is ${:,.2f}",
                    desc,
                    order.totalQuantity,
                    multiplier,
                    amt,
                    leverageKind,
                    amt * multiplier * order.totalQuantity,
                )

        # also print a delta-adjusted leverage for the underlying if delta is less than 1
        symkey = lookupKey(contract)
        highCommission = status.maxCommission or status.commission

        if highCommission and (ticker := self.quoteState.get(symkey)):
            if ticker.modelGreeks and abs(ticker.modelGreeks.delta) < 1:
                for amt in (1, 3, 9):
                    # NOTE: we report both the value move and the exit-at profit AFTER COMISSIONS assuming the 'highComission' commission is 2x (open+close)
                    # Also note: this is an estimate, because we are not adjusting for delta growing as price moves in our favor.
                    # Also also note: this move leverage can be misleadig for volatility straddles because we expect vega thus gamma to overwhelm underlying movements directly.
                    move = (
                        amt
                        * multiplier
                        * abs(order.totalQuantity)
                        * abs(ticker.modelGreeks.delta)
                    )
                    logger.info(
                        "[{}] ${:,.2f} UNDERLYING MOVE LEVERAGE is {} (exit: {:>6})",
                        desc,
                        amt,
                        fmtmoney(move),
                        fmtmoney(move - (highCommission * 2)),
                    )

        # "MAIN" for "MAINTENANCE" to match the length of "INIT" above for alignment.

        # if we have any commission, estimate how many leveraged points we need to "Earn it out" for a round trip.
        if status.minCommission or status.commission:
            low = status.minCommission or status.commission
            high = status.maxCommission or status.commission

            # only print number of commissions reported (some products have "flat" commissions and the price is always the same,
            # so for those cases we never have a low/high to report, it's just one consistent nubmer)
            l = 2 * low
            h = 2 * high
            l2 = (2 * low) / multiplier / abs(order.totalQuantity)
            h2 = (2 * high) / multiplier / abs(order.totalQuantity)

            if low == high:
                # if low commission == high commission, then we only have one commission to report.
                points = f"(${l:,.{digits}f}): ${l2:,.{digits}f}"
            else:
                points = f"(${l:,.{digits}f} to ${h:,.{digits}f}): ${l2:,.{digits}f} to ${h2:,.{digits}f}"

            logger.info(
                "[{}] CONTRACT POINTS TO PAY ROUNDTRIP COMMISSION {}",
                desc,
                points,
            )

        if status.minCommission:
            # options and stocks have a range of commissions
            logger.info(
                "[{}] PREVIEW COMMISSION PER {}: ${:.4f} to ${:.4f}",
                desc,
                nameOfThing,
                status.minCommission / order.totalQuantity,
                status.maxCommission / order.totalQuantity,
            )

            if multiplier > 1:
                # (Basically: how much must the underlying change in price for you to pay off the commission for this order.
                tcmin = status.minCommission / order.totalQuantity / multiplier
                tcmax = status.maxCommission / order.totalQuantity / multiplier
                logger.info(
                    "[{}] PREVIEW COMMISSION PER UNDERLYING: ${:.4f} to ${:.4f} (2x: ${:.4f} to ${:.4f})",
                    desc,
                    tcmin,
                    tcmax,
                    2 * tcmin,
                    2 * tcmax,
                )
        elif status.commission:
            # futures contracts and index options contracts have fixed priced commissions so
            # they don't provide a min/max range, it's just one guaranteed value.
            legComm = ""
            if isinstance(contract, Bag):
                legComm = f" (per ech leg ({len(contract.comboLegs)}): ${status.commission / (order.totalQuantity * len(contract.comboLegs)):.4f})"

            logger.info(
                "[{}] PREVIEW COMMISSION PER CONTRACT: ${:.4f}{}",
                desc,
                (status.commission) / order.totalQuantity,
                legComm,
            )

            tc = status.commission / order.totalQuantity / multiplier
            if multiplier > 1:
                logger.info(
                    "[{}] PREVIEW COMMISSION PER UNDERLYING: ${:.4f} (2x: ${:.4f})",
                    desc,
                    tc,
                    2 * tc,
                )

        # we allow bid to be none if this is a 'buy on unlikely' scenario
        # assert bid is not None

        if bid is None:
            bid = 0

        assert ask is not None

        # calculate percentage width of the spread just to note if we are trading difficult to close positions
        # Note: if bid is '0' (for some spreads) then we just use "whole quantity" for the divisor percentage math here.
        spreadDiff = ((ask - bid) / max(1, bid)) * 100
        logger.info(
            "[{}] BID/ASK SPREAD IS {:,.2f} % WIDE ({} spread @ {} total)",
            desc,
            spreadDiff,
            fmtmoney(ask - bid),
            fmtmoney(
                order.totalQuantity
                * float(previewPrice)
                * (float(spreadDiff) / 100)
                * multiplier
            ),
        )

        if spreadDiff > 5:
            logger.warning(
                "[{}] WARNING: BID/ASK SPREAD ({:,.2f} %) MAY CAUSE NOTICEABLE LOSS/SLIPPAGE ON EXIT",
                desc,
                spreadDiff,
            )

        # TODO: make this delta range configurable? config file? env? global setting?
        if isinstance(contract, (Option, FuturesOption)):
            mg = self.quoteState[symname].modelGreeks
            if mg:
                delta = mg.delta
                if not delta:
                    logger.warning("[{}] WARNING: OPTION DELTA NOT POPULATED YET", desc)
                elif abs(delta) <= 0.15:
                    logger.warning(
                        "[{}] WARNING: OPTION DELTA IS LOW ({:.2f}) — THIS MAY NOT WORK FOR SHORT TERM TRADING",
                        desc,
                        delta,
                    )
            else:
                logger.warning(
                    "[{}] WARNING: OPTION GREEKS NOT YET POPULATED FOR DELTA CHECKING",
                    desc,
                )

        # if this trade INCREASES our equity, let's see if there's a risk involved
        # Note: this is primarily useful for credit spreads where we receive a credit for a fixed margin risk.
        # TODO: also run R:R calculations when buying debit spreads versus the spread filling to 100%
        # TODO: test this against regular equity shorts to see what it reports too.
        # TODO: this doesn't run properly if there's zero margin impact (due to other offsets) but it's a new short position.
        # TODO: maybe only run this if the contract is a Bag instead of checking equity loan change as the trigger?
        # This needs to account for reg-t shorts (initial == maint then credit received as EWLC) and also
        # account for SPAN shorts (initial > maint and credit received not reflected in EWLC).
        # For Reg-T shorts, the account holds the full short block amount as initial margin because
        # the credit can't relieve your own margin (so if you short sell $20k on a $20k risk, you still have $20k margin).
        # For SPAN shorts, the exact risk is _only_ the margin change (so if you
        # short sell $20k on a $20k risk, you have $0 margin chnage since the credit cancels out the risk).
        if status.initMarginChange > 0 and order.action == "SELL":
            ewlc = status.equityWithLoanChange
            if status.initMarginChange > 0:
                # If equity is increasing, then this is a short (receiving credit) with margin risk.
                # Our risk is (total increase in stop-out margin call requirement).
                # Maint Margin is always less than or equal to initial margin, so it will stop-out the trade at an
                # equal or _sooner_ level than the initial margin requirement.
                risk = status.maintMarginChange - ewlc
                riskPct = risk / (baseTotal or risk)

                # the more decimals the more extreme the ratio generated, so instead of 69:100 at 2 decimals, show 7:10 at 1 decimal.
                fracMin = Fraction(round(riskPct, 1)).limit_denominator()
                riskMin, rewardMin = fracMin.numerator, fracMin.denominator

                # also provide an even lower resoultion R:R single digit ratio
                fracMin2 = Fraction(round(riskPct)).limit_denominator()
                riskMin2, rewardMin2 = fracMin2.numerator, fracMin2.denominator
                logger.warning(
                    "[{}] RISKING MARGIN: ${:,.2f} (received ${:,.2f} credit; risking {:,.2f}x == {}:{} Risk:Reward ratio{})",
                    desc,
                    risk,
                    baseTotal,
                    riskPct,
                    riskMin,
                    rewardMin,
                    # provide a lower resolution (but easier to read) ratio if the first attempt isn't 1:N already:
                    f" ({riskMin2}:{rewardMin2})"
                    if riskMin > 10 and (riskMin, rewardMin) != (riskMin2, rewardMin2)
                    else "",
                )

        # (if trade isn't valid, trade is an empty list, so only print valid objects...)
        if status:
            if not (status.commission or status.minCommission or status.maxCommission):
                logger.error(
                    "[{}] TRADE NOT VIABLE DUE TO MISSING COMMISSION ESTIMATES",
                    desc,
                )

            # excess = self.accountStatus["ExcessLiquidity"] - status.initMarginChange
            excess = status.equityWithLoanAfter - status.initMarginAfter
            if excess < 0:
                logger.warning(
                    "[{}] TRADE NOT VIABLE. MISSING EQUITY: {}",
                    desc,
                    fmtmoney(excess),
                )
            else:
                # show rough estimate of how much we're spending.
                # for equity instruments with margin, we use the margin buy requirement as the cost estimate.
                # for non-equity (options) without margin, we use the absolute value of the buying power drawdown for the purchase.
                # TODO: this value is somewhere between wrong or excessive if there's already marginable positions engaged since
                #       the calculation here is assuming a new position request is the only position in the account.

                # amount required to hold trade - amount extraced from trade
                # options are: 0 - (-premium)
                # short spreads are: full margin - (received credit)
                # longs are: (margin holding) - (almost nothing (basically just commission offsets))
                # Note: assuming standard reg-t margin, your short spreads are NOT returned to you as buying power,
                #       so we DO NOT add credits from spreads back as a risk minimization here.
                #       e.g. if you have a $5,000 margin spread with a +$3,000 credit, you still have $5,000 reduced
                #            buying power instead of (trade.initMarginChange - trade.equityWithLoanChange = $5,000 - $3,000) reduced buying power
                # UPDATE: for credit events, we assume the initial margin is our cost impact.
                #         for debit events, we assume there is no margin change, but the EWL is negative (cost of position) so we flip it as our risk
                risk = status.maintMarginChange or -status.equityWithLoanChange

                # there is also the account status 'InitMarginReq' field we could potentially use as well.
                logger.info(
                    "[{}] PREVIEW REMAINING INIT CASH AFTER TRADE ({}): {}",
                    desc,
                    fmtmoney(-status.initMarginChange),
                    fmtmoney(status.equityWithLoanAfter - status.initMarginAfter),
                )

                logger.info(
                    "[{}] PREVIEW REMAINING MAINT CASH AFTER TRADE ({}): {}",
                    desc,
                    fmtmoney(-status.maintMarginChange),
                    fmtmoney(status.equityWithLoanAfter - status.maintMarginAfter),
                )

                fundsDiff = (risk / self.accountStatus["AvailableFunds"]) * 100

                if fundsDiff < 0:
                    # your account value is GROWING, this is a funds increase
                    logger.info(
                        "[{}] PREVIEW TRADE PERCENTAGE OF FUNDS ADDED: {:,.2f} %",
                        desc,
                        -fundsDiff,
                    )
                else:
                    # else, account value is LOWERING so this is a funds reduction
                    logger.info(
                        "[{}] PREVIEW TRADE PERCENTAGE OF FUNDS USED: {:,.2f} %",
                        desc,
                        fundsDiff,
                    )

    def ifthenAbs(self, content: str):
        """Just get an absolute value here..."""
        return abs(float(content))

    def ifthenQuantityForContract(self, symbol: str):
        """Convert input symbol to contract then look up quantity.

        Note: since this is a *non-async* ifthen function, we can just return the result directly."""
        contract = contractForName(symbol)
        found = self.quantityForContract(contract)
        return found

    async def ifthenExtensionVerticalSpreadCall(
        self, mailbox: dict[str, Any], symbol: str, startStrike: float, distance: float
    ):
        return await self.ifthenExtensionVerticalSpread(
            "c", mailbox, symbol, startStrike, distance
        )

    async def ifthenExtensionVerticalSpreadPut(
        self, mailbox: dict[str, Any], symbol: str, startStrike: float, distance: float
    ):
        return await self.ifthenExtensionVerticalSpread(
            "p", mailbox, symbol, startStrike, distance
        )

    async def ifthenExtensionVerticalSpread(
        self,
        side: Literal["p", "c"],
        mailbox: dict[str, Any],
        symbol: str,
        startStrike: float,
        distance: float,
    ):
        """Calculate a vertical spread for symbol and also subscribe to spread quote for live price updating."""
        isq = lang.IOpStraddleQuote(
            state=self,
            symbol=symbol,
            overrideATM=startStrike,
            widths=f"v {side} 0 {distance}".split(),
        )

        contracts = await isq.run()

        assert (
            len(contracts) == 3
        ), f"Expected 3 results from spread adding, but got {len(contracts)=} for {contracts=}?"

        # technically, we think these are always in the same order of [bag, buy, sell], but if the order
        # changes, we want to always grab the bag first to populate legIds, then use legIds again.
        # Overall, the 'contracts' is only 3 elements so looping twice isn't a problem.
        for c in contracts:
            if isinstance(c, Bag):
                bag = c
                legIds = {leg.conId: leg.action for leg in bag.comboLegs}

        for c in contracts:
            if lid := legIds.get(c.conId):
                if lid == "BUY":
                    buyLeg = c
                elif lid == "SELL":
                    sellLeg = c

        mailbox["result"] = True
        mailbox["spread"] = self.nameForContract(bag)
        mailbox["buy"] = self.nameForContract(buyLeg)
        mailbox["sell"] = self.nameForContract(sellLeg)
        mailbox["contract"] = bag
        mailbox["contractBuy"] = buyLeg
        mailbox["contractSell"] = sellLeg

    def nameForContract(self, contract: Contract) -> str:
        return nameForContract(contract, self.conIdCache)

    @property
    def positionsDB(self) -> dict[int, Position]:
        """Return a dict holding a mapping of contract ids to contracts.

        This currently uses some ib_async internals which we should have better insight to.
        """
        positionsDB = self.ib.wrapper.positions[self.accountId]
        return positionsDB

    def quantityForContract(self, contract: Contract) -> float:
        """Returns positive numbers for longs, negative counts for shorts, and 0 for no position."""

        if pos := self.positionsDB.get(contract.conId):
            pos.position

        return 0

    def averagePriceForContract(self, contract: Contract) -> float:
        """Using live portfolio data, return current average cost for position having contract ids.

        We accept multiple contract ids to allow for this to generate the combined cost basis for spreads too."""

        cost = 0.0

        # fetch all relevant contract ids for the contract provided
        # (e.g. spreads/bags have inner contract ids, but everything else is just a single top-level conId)
        # (also ignoring type here because we know each of these elements has .conId even though they are different types)
        src = contract.comboLegs if contract.comboLegs else [contract]  # type: ignore
        conIds = [x.conId for x in src]

        positionsDB = self.positionsDB
        # TODO: update ib_async API to allow direct retrieval of portfolio or positions by contract id instead of fetching all then iterating.
        for conId in conIds:
            # FIX: using internal API hack to get direct access to contract-level account row fetching
            row = positionsDB.get(conId)

            # missing rows is the same as a $0 cost basis because we don't hold the position.
            # (the math still works out; if going short new entry is < 0; if going long, new entry is > 0)
            if row is None:
                return 0

            # API TRICK: "position" objects have 'avgCost' while "portfolio" objects have 'averageCost'
            cost += row.avgCost

        return cost

    def multiplier(self, contract: Contract) -> float | int:
        """Abstraction to get the multiplier for any contract.

        Why do we need this?

        Equity symbols have no multipler, so we use 1.
        Options and Futures have a definied multiplier.
        Bags / Spreads have no multiplier defined at the contract leve, so we need to look _inside_ the bag to find the multiplier.
        """
        if isinstance(contract, (Option, FuturesOption)):
            mul = float(contract.multiplier or 1.0)
        elif isinstance(contract, Bag):
            # steal multiplier of first thing in contract. we assume it's okay? This would be wrong for buy-write bags and is only valid for spreads.
            innerContract = self.conIdCache.get(contract.comboLegs[0].conId)
            mul = float(innerContract.multiplier or 1.0)
        else:
            mul = float(contract.multiplier or 1.0)

        if mul == (imul := int(mul)):
            mul = imul

        return mul

    def quantityForAmount(
        self,
        contract: Contract,
        amount: Decimal | int | float,
        limitPrice: Decimal | float,
    ) -> int | float:
        """Return valid quantity for contract using total dollar amount 'amount'.

        Also compensates for limitPrice being a contract quantity.

        Also compensates for contracts allowing fractional quantities (Crypto)
        versus only integer quantities (everything else)."""

        # For options, the multipler is PART OF THE COST OF BUYING because a $0.15 option costs $15 to buy,
        # but for futures, the multiplier is NOT PART OF THE BUY COST because buying futures only costs
        # future margin which is much less than the quoted contract price (but the futures margin is
        # technically aorund 4% of the total value because a $4,000 MES contract has a 5 multipler so
        # your $4,000 MES contract is holding $20,000 notional on a $1,700 margin requirement).
        mul = self.multiplier(contract)

        assert mul > 0

        # total spend amount divided by price of thing to buy == how many things to buy
        # (rounding to fix IBKR error for fractional qty: "TotalQuantity field cannot contain more than 8 decimals")
        qty = float(amount) / (float(limitPrice) * float(mul))
        if qty <= 0:
            logger.error(
                "Sorry, your calculated quantity is {:,.f} so we can't order anything!",
                qty,
            )
            return 0

        if isinstance(contract, Crypto):
            # only crypto orders support fractional quantities over the API.
            qty = round(qty, 8)
        else:
            # TODO: if IBKR ever enables fractional shares over the API,
            #       we can make the above Crypto check for (Crypto, Stock).
            qty = math.floor(qty)

        return qty

    def orderPriceForSpread(self, contracts: Sequence[Contract], positionSize: int):
        """Given a set of contracts, attempt to find the closing order."""
        ot = self.ib.openTrades()

        contractIds = set([c.conId for c in contracts])
        # Use a list so we can collect multiple exit points for the same position.
        ts = []
        for t in ot:
            if not isinstance(t.contract, Bag):
                continue

            legIds = set([c.conId for c in t.contract.comboLegs])
            if legIds == contractIds:
                qty, price = t.orderStatus.remaining, t.order.lmtPrice
                ts.append((qty, price))

        # if only one and it's the full position, return without formatting
        if len(ts) == 1:
            if abs(int(positionSize)) == abs(ts[0][0]):
                return ts[0][1]

        # else, break out by order size, sorted from smallest to largest exit prices
        return sorted(ts, key=lambda x: abs(x[1]))

    def orderPriceForContract(self, contract: Contract, positionSize: float | int):
        """Attempt to match an active closing order to an open position.

        Works for both total quantity closing and partial scale-out closing."""
        ot = self.ib.openTrades()

        # Use a list so we can collect multiple exit points for the same position.
        ts = []
        for t in ot:
            # t.order.action is "BUY" or "SELL"
            opposite = "SELL" if positionSize > 0 else "BUY"
            if (
                t.order.action == opposite
                and t.contract.localSymbol == contract.localSymbol
            ):
                # Closing price is opposite sign of the holding quantity.
                # (i.e. LONG positions are closed for a CREDIT (-) and
                #       SHORT positions are closed for a DEBIT (+))
                assert (
                    t.order.lmtPrice is not None
                ), "How is the order limit price None here?"

                ts.append(
                    (
                        int(t.orderStatus.remaining),
                        float(
                            math.copysign(1, positionSize)
                            * -1
                            * float(t.order.lmtPrice)
                        ),
                    )
                )

        # if only one and it's the full position, return without formatting
        if len(ts) == 1:
            if abs(int(positionSize)) == abs(ts[0][0]):
                return ts[0][1]

        # else, break out by order size, sorted from smallest to largest exit prices
        return sorted(ts, key=lambda x: abs(x[1]))

    def currentQuote(self, sym, show=True) -> tuple[float | None, float | None]:
        # TODO: maybe we should refactor this to only accept qualified contracts as input (instead of string symbol names) to avoid naming confusion?
        q = self.quoteState.get(sym)

        # if quote did not exist, then we need to check it...
        assert q and q.contract, f"Why doesn't {sym} exist in the quote state?"

        # use our potentially synthetically-derived live quotes for spreads (if IBKR isn't quoting us bid/ask for some reason)
        current = q.quote()

        # (we use 'is not None' here because with spreads, a bid or ask of $0 _is_ valid for credit spreads)
        hasBid = current.bid is not None
        hasAsk = current.ask is not None

        hasQuotes = hasBid or hasAsk

        # only optionally print the quote because printing technically requires extra time
        # for all the formatting and display output
        if show and hasQuotes:
            ago = (
                "now"
                if q.time >= self.nowpy
                else as_duration((self.nowpy - (q.time or self.nowpy)).total_seconds())
            )

            if q.lastTimestamp:
                agoLastTrade = (
                    "now"
                    if q.lastTimestamp >= self.nowpy
                    else as_duration((self.nowpy - q.lastTimestamp).total_seconds())
                )
            else:
                agoLastTrade = "never received"

            digits = self.decimals(q.contract)

            assert current.bid is not None or current.ask is not None

            # if no bid, just use ask or last, whichever exists first
            if not hasBid:
                mid = current.ask or q.last
            else:
                # another redundant check to help the type checker stop complaining
                assert current.bid is not None and current.ask is not None
                mid = (current.bid + current.ask) / 2

            # don't pass zero bid/ask calculation to the DB or else it gets upset
            # NOTE: If there is zero tick width between bid/ask (e.g. $0.80 bid, $0.85 ask on a $0.05 tick),
            #       then the midpoint will be either the bid or the ask directly because it can't actually
            #       divide them in half.
            if mid:
                mid = self.idb.roundOrNothing(q.contract, mid) or mid

            # fmt: off
            show = [
                f"{q.contract.secType} :: {q.contract.localSymbol or q.contract.symbol}:",
                f"bid {current.bid:,.{digits}f} x {current.bidSize}" if q.bid else "bid NONE",
                f"mid {mid:,.{digits}f}",
                f"ask {current.ask:,.{digits}f} x {current.askSize}",
                f"last {q.last:,.{digits}f} x {q.lastSize}" if q.last else "(no last trade)",
                f"last trade {agoLastTrade}" if q.lastTimestamp else "(no last trade timestamp)",
                f"updated {ago}" if q.time else "(no last update time)",
            ]
            # fmt: on
            logger.opt(depth=1).info("    ".join(show))

        # updated price picking logic: if we have a live bid/ask, return them.
        # else, if we don't have a bid/ask, use the last reported price (if it exists).
        # else else, return nothing because there's no actual price we can read anywhere.

        # if no quote yet (or no prices available), return last seen price...
        if hasQuotes:
            return current.bid, current.ask

        # if last exists, use it.
        if q.last is not None:
            last = q.last
            return last, last

        # else, we found no valid price option here.
        return None, None

    async def loadExecutions(self) -> None:
        """Manually fetch all executions from the gateway.

        The IBKR API only sends live push updates for executions on the _current_ client,
        so to see executions from either _all_ clients or executions before this client started,
        we need to ask for them all again.
        """

        logger.info("Fetching full execution history...")
        try:
            # manually flag "we are loading historical commissions, so don't run the event handler"
            self.loadingCommissions = True

            with Timer("Fetched execution history"):
                try:
                    await asyncio.wait_for(self.ib.reqExecutionsAsync(), 7)
                except:
                    logger.error("Executions failed to load before the timeout period.")
        finally:
            # allow the commission report event handler to run again
            self.loadingCommissions = False

    def updateOrder(self, trade: Trade):
        # Only print update if this is regular runtime and not
        # the "load all trades on startup" cycle
        if not self.connected:
            return

        logger.warning(
            "[{} :: {} :: {}] Order update: {}",
            trade.orderStatus.orderId,
            trade.orderStatus.status,
            trade.contract.localSymbol,
            trade,
        )

        # notify any watchers only when order is 100% filled
        if trade.log[-1].status == "Filled" and trade.orderStatus.remaining == 0:
            f = self.fillers[trade.contract]
            f.trade = trade
            f.set()

    def errorHandler(self, reqId, errorCode, errorString, contract):
        # Official error code list:
        # https://interactivebrokers.github.io/tws-api/message_codes.html
        # (note: not all message codes are listed in their API docs, of course)
        if errorCode in {1102, 2104, 2108, 2106, 2107, 2119, 2152, 2158}:
            # non-error status codes on startup or informational messages during running.
            # we ignore reqId here because it is either always -1 or a data request id (but never an order id)
            logger.info(
                "API Status [code {}]: {}",
                errorCode,
                errorString,
            )
        else:
            # Instead of printing these errors directly, we pass them through a deduplication
            # filter because sometimes we get unlimited repeated error messages (which aren't
            # actually errors) and we want to suppress them to only one repeated error update
            # every 30 seconds instead of 1 error per second continuously.
            msg = "{} [code {}]: {}{}".format(
                f"Order Error [orderId {reqId}]"
                if (reqId > 0 and errorCode not in {321, 366})
                else "API Error",
                errorCode,
                readableHTML(errorString),
                f" for {contract}" if contract else "",
            )

            self.duplicateMessageHandler.handle_message(
                message=msg, log_func=logger.error
            )

    def cancelHandler(self, err):
        logger.warning("Order canceled: {}", err)

    def commissionHandler(self, trade, fill, report):
        # Only report commissions if not bulk loading them as a refresh
        # (the bulk load API causes the event handler to fire for each historical fill)
        if self.loadingCommissions:
            logger.warning(
                "Ignoring commission because bulk loading history: [{:>2} :: {} {:>7.2f} of {:>7.2f} :: {}]",
                fill.execution.clientId,
                fill.execution.side,
                fill.execution.shares,
                fill.execution.cumQty,
                fill.contract.localSymbol,
            )
            return

        # TODO: different sounds if PNL is a loss?
        #       different sounds for big wins vs. big losses?
        #       different sounds for commission credit vs. large commission fee?
        # TODO: disable audio for algo trades?

        if self.speak.url:
            # using "BOT" and "SLD" as real words because the text-to-speech was pronouncing "SLD" as individual letters "S-L-D"
            side = "bought" if fill.execution.side == "BOT" else "sold"

            fillQty = f"{fill.contract.localSymbol} ({side} {int(fill.execution.shares)} (for {int(fill.execution.cumQty)} of {int(trade.order.totalQuantity)}))"

            #  This triggers on a successful close of a position (TODO: need to fill out more details)
            if fill.commissionReport.realizedPNL:
                PorL = "profit" if fill.commissionReport.realizedPNL >= 0 else "loss"

                content = f"CLOSED: {trade.orderStatus.status} FOR {fillQty} ({PorL} ${round(fill.commissionReport.realizedPNL, 2):,})"
            else:
                # We notify about orders HERE instead of in 'orderExecuteHandler()' because HERE we have details about filled/canceled for
                # the status, where 'orderExecuteHandler()' always just has status of "Submitted" when an execution happens (also with no price details) which isn't as useful.
                content = f"OPENED: {trade.orderStatus.status} FOR {fillQty} (commission {fmtmoney(fill.commissionReport.commission)})"

            self.task_create(content, self.speak.say(say=content))

        logger.warning(
            "[{} :: {} :: {}] Order {} commission: {} {} {} at ${:,.2f} (total {} of {}) (commission {} ({} each)){}",
            trade.orderStatus.orderId,
            trade.orderStatus.status,
            trade.contract.localSymbol,
            fill.execution.orderId,
            fill.execution.side,
            fill.execution.shares,
            fill.contract.localSymbol,
            fill.execution.price,
            fill.execution.cumQty,
            trade.order.totalQuantity,
            fmtmoney(fill.commissionReport.commission),
            fmtmoney(fill.commissionReport.commission / fill.execution.shares),
            f" (pnl {fill.commissionReport.realizedPNL:,.2f})"
            if fill.commissionReport.realizedPNL
            else "",
        )

        self.updateAgentAccountStatus(
            "commission",
            FillReport(
                orderId=trade.orderStatus.orderId,
                conId=trade.contract.conId,
                sym=trade.contract.localSymbol.replace(" ", ""),
                side=fill.execution.side,
                shares=fill.execution.shares,
                price=fill.execution.price,
                pnl=fill.commissionReport.realizedPNL,
                commission=fill.commissionReport.commission,
                when=fill.execution.time,
            ),
        )

    def newsBHandler(self, news: NewsBulletin):
        logger.warning("News Bulletin: {}", readableHTML(news.message))

    def newsTHandler(self, news: NewsTick):
        logger.warning("News Tick: {}", news)

    async def orderExecuteHandler(self, trade, fill):
        isBag: Final = isinstance(trade.contract, Bag)
        logger.warning(
            "[{} :: {}] Trade executed for {}",
            trade.orderStatus.orderId,
            trade.orderStatus.status,
            self.nameForContract(trade.contract),
        )

        if isBag:
            # if order executed as a bag, attach each contract id to the same spread
            # (but only create if it doesn't already exist; we don't want partial fills replacing exsiting positions)
            newPosition = IPosition(trade.contract)
            for leg in trade.contract.comboLegs:
                if leg.conId not in self.iposition:
                    self.iposition[leg.conId] = newPosition
        else:
            conId = trade.contract.conId
            if conId not in self.iposition:
                self.iposition[conId] = IPosition(trade.contract)

    def positionEventHandler(self, position: Position):
        """Update position against our local metadata.

        Note: positions are always SINGLE contracts (i.e. you will never get a Bag contract here).
        """

        # TODO: re-evaluate if we actually need this? It doesn't work on startup since we moved
        #       this here instead of in the order notification system. Maybe these _are_ subscribed
        #       on startup by default?
        if False:
            conId = position.contract.conId
            if conId not in self.pnlSingle:
                self.pnlSingle[conId] = self.ib.reqPnLSingle(self.accountId, "", conId)
            else:
                self.iposition[position.contract].update(
                    position.contract, position.position, position.avgCost
                )

                # if quantity is gone, stop listening for updates and remove.
                if position.position == 0 and conId in self.pnlSingle:
                    self.ib.cancelPnLSingle(self.accountId, "", conId)
                    del self.pnlSingle[conId]

    async def positionActiveLifecycleDoctrine(
        self,
        contract: Contract,
        target: OrderIntent,
        upTick: float | Decimal = 0.25,
        downTick: float | Decimal = 0.25,
    ):
        """Begin an acquisition or distribution run for 'position' up to 'target' quantity or total spend.

        'contract' is the instrument for this order.
        'target' is the price and bracket description for this order.
        'upTick' is the next scale-in offset from the previously executed price (higher).
        'downTick' is the next scale-in offset from the previously executed price (lower).

        This is essentialy a meta-wrapper around the 'buy' automation because:
            - 'buy' already loops until it purchases the target quantity (or price)
            - 'buy' already auto-adjusts price (but only to be MORE towards a live price)

        Here we just want to puppet 'buy' a little looser to:
            - buy immediately
            - wait for price to move up or down
            - buy more
            - wait for price to move again, continue buying, until total quantity reached
            - then camp at a stop loss or take profit

        Basically, we want to avoid doing a single large purchase when price is already floating in a ± 10% range sometimes.
        We would rather buy at X, X+N, X-N, X, X+N+K, X-N+K etc for a more realistic time-average cost basis. Also, if the price
        _temporarily_ goes against us, we can acquire at a better cost basis, but if it _continues_ going agianst us, we can stop out.

        Essentially: we want to optimize our chances of staying in a +EV position without being afraid of buying and having something
        retrace to an extent we feel we need to bounce out, only to have it reverse up again.

        Also, if we are in a clear trend direction, it can be safer to scale in at N, N+K, N+K+Z, N+K+Z+Y at worse and worse cost basis
        because prices are working for us. I would rather have purchases between $4 and $10 for something going to $30 than being too afraid
        of buying all at $4 because it could reverse or all at $5 because it could reverse, or buying all at $10 and then having it actually reverse, etc.
        """
        ...

        # create our position tracker from the input contract.
        # Details of live executions will be updated in this position object by the execution event handlers.
        position: Final = IPosition(contract)
        self.iposition[contract] = position

        f: Final = self.fillers[contract]
        name: Final = self.nameForContract(contract)

        isPreview: Final = target.preview

        assert target.qty

        # yeah, a dreaded while-true..... sorry.
        # Basically, our initial condition _can_ be None, but by None we mean "please wait and try again" not "STOP WE ARE DONE,"
        # so we need a way to capture "None" and retry again instead of doing "while not complete..." because complete can be None
        # as a valid indicator too.
        prevCompletePct = None
        while True:
            while (completePct := position.percentComplete(target)) is None:
                # if percentage report is None, it means the IPosition is in the middle of an async update,
                # so we need to WAIT FOR MORE DATA before we have a valid reading from the percentage output.
                logger.warning(
                    "[{}] Waiting for IPosition to complete its updates...",
                    contract.localSymbol,
                )
                await asyncio.sleep(0.001)

            # we need to verify the number actually changed (maybe we are reading too soon after an order completion
            # and the position update callbacks haven't fired to change the position values yet, but we need _new_ details
            # to continue properly).
            if prevCompletePct is not None:
                # if previous is the same as current, no changes were found, so we need to loop again...
                if prevCompletePct == completePct:
                    await asyncio.sleep(0.001)
                    continue

            prevCompletePct = completePct

            # since we got a non-None result as completePct, the remaining 'position' members should be populated

            # if percentage of completness is reported as 100% or larger, we are done here.
            if completePct >= 1:
                logger.info(
                    "[{}] Goal is reached. No more active trade management for purchasing!",
                    contract.localSymbol,
                )
                break

            # if we have ZERO percentage complete so far: buy now and try again at next bounds
            if completePct == 0:
                # TODO: adaptive (or parameter) start quantity
                startQty = 2
                algo = "AF"
                cmd = f"buy '{name}' {startQty} {algo}"

                if isPreview:
                    logger.info("[preview :: {}] Would have run: {}", name, cmd)
                    break
                else:
                    logger.info("[{}] Running: {}", name, cmd)
                    await self.buildAndRun(cmd)

                    await f.orderComplete()
                    continue

            # remaining percentage to acquire...
            # Currently, we are targeting a maximum of 5% purchase per buy (or just the remaining quantity, whichever is less)
            remainingPct = min(0.05, 1 - completePct)

            # get STRING representation of quantity to buy (e.g. short is '-1', short cash is '-$100', etc)
            buyVal: str = target.qtyPercent(remainingPct).nice()

            # If remaining percent is _more than_ the remaining quantity, just add the missing quantity.
            if isinstance(target.qty, DecimalCash):
                # if quantity is cash, we want to use cash-based differences of current spend
                # (either: buy buyVal or buy REMAINING quantity if remaining is less than buy val)
                remainingQty = min(
                    target.qty - Decimal(str(position.totalSpend)), buyVal
                )
            else:
                # else, is shares, so use direct quantity instead of price data
                remainingQty = min(target.qty - Decimal(str(position.totalQty)), buyVal)

            # TODO: create dynamic metric for deciding how much to order based on:
            #   - volatility
            #   - "safety net" of average cost versus current market price
            #   - aggressiveness score?
            #   - time of day
            #   - active current and historical IV of contract?

            # else, acquisition has started but is not complete, so we need to continue scheduling order placement triggers.
            # GOAL: buy LOW, buy HIGH up to QTY, when QTY complete, enact STOP or PROFIT conditions.

            assert f.trade
            lastPrice = f.trade.orderStatus.avgFillPrice
            assert lastPrice

            # buy a SMALLER LEVEL or HIGHER LEVEL
            # TODO: evaluate other metric guards for entry besides price ticks? prevscores? delta growth?
            # TODO: more easily adjustable price ticks (use scale system of order intent?)
            # TODO: how to cancel/abnadon predicates once generated if we want to cancel this completely?
            dualside = f"if ('{name}' mid <= {lastPrice} - {downTick}) or ('{name}' mid >= {lastPrice} + {upTick}): buy '{name}' {str(remainingQty)} AF"

            # submit ifthen predicate then WAIT FOR IT TO FILL
            logger.info(
                "[{}] Building predicate for next acquisition: {}", name, dualside
            )

            if not isPreview:
                predicateId = await self.buildAndRun(dualside)

                # TODO: how to error check here if we run the buy, but the buy errors out? We would just wait forever here because the fill will never happen.
                #       Do we also have to wait on the _status_ of the buy command after the predicate is executes somehow? We would need "predicate executed" trigger
                #       to start a countdown/timeout waiting for the buy to fill..."

                await f.orderComplete()

        # TODO: integrate this with the OrderIntent scale system?
        # TODO: time-of-day blocking for these (follow overnight volstops?)
        # TODO: trigger on algo/volstop change to true?
        # WHEN ALL ACQUIRED, ENACT STOP LOSS AND TAKE PROFIT EXIT CONDITIONS
        closeQty = position.closeQty
        assert closeQty

        # TODO: we should be assembling these as a proper Peers/OCA object so we can
        #       easily pick between using one or both.
        if target.bracketLoss:
            if target.isBracketLossPercent:
                stopLoss = position.closePercent(-float(target.bracketLoss) / 100)
            else:
                stopLoss = position.closeCash(-float(target.bracketLoss))

        if target.bracketProfit:
            if target.isBracketProfitPercent:
                takeProfit = position.closePercent(float(target.bracketProfit) / 100)
            else:
                takeProfit = position.closeCash(float(target.bracketProfit))

        # TODO: for closing out, we could do the buy automation *or* we could run THIS ENTIRE PROCESS AGAIN but
        #       with a close target instead of the open target... (then obviously don't loop again at the end).
        # TODO: should also adjust 'mid' to be different sides for long/short stop/profit.
        close = f"if '{name}' {{ mid <= {stopLoss} or mid >= {takeProfit} }}: buy '{name}' {closeQty} AF"
        logger.info("[{}] Building predicate for exit: {}", name, close)

        if not isPreview:
            predicateId = await self.buildAndRun(close)

    async def predicateSetup(self, prepredicate: ifthen.CheckableRuntime):
        """Attach data extractors and custom functions to all predicates inside 'prepredicate'.

        The ifthen predicate language only _describes_ what to check, but we must provide the predicate(s)
        with the actual data sources and custom functions so the predicate(s) can execute their checks
        with live data on every update.
        """

        # Now we need to traverse the entire predicate condition hierarchy so we can bind
        # individual DataExtractor instances to live data elements to use for value extraction.

        symbolToTickerMap: dict[Hashable, ITicker] = {}

        pid = prepredicate.id

        # a CheckableRuntime predicate condition may have multiple inner predicates, so we must
        # introspect all _inner_ predicates to properly attach their data accessors.
        for predicate in prepredicate.actives:
            extractedSymbols = set()

            # Upon return of PARSED result, we need to:
            #  For each symbol in parsed.symbols, get contracts via: await self.state.positionalQuoteRepopulate(sym)
            # We need to pass in a unified 'datasource' capable of looking up (symbol (string or contract id or ticker (?????)), FIELD (SMA), TIMEFRAME (35s), SUBFIELD (5 lookback or 10 lookback))
            #  - though, instead of needing string/id/ticker binding, we attach a custom closure already attached to the target symbol so it is always bound to the correct data source provider itself.
            #     - but then we have to pick between: Are we looking up a field on the ITicker or a field in the Algo Feed?
            #        - for Algo Feed, we also then need to save each {Symbol -> {Duration -> {algo -> {result: value}}}} and repopulate {Symbol -> {Duration...}} when a new Duration change is received.
            for symbol in predicate.actuals:
                assert isinstance(symbol, str)
                foundsym, c = await self.positionalQuoteRepopulate(symbol)

                logger.info("[{}] Tracking contract: {}", pid, c)

                # subscribe if not subscribed (no-op if already subscribed, but returns symkey either way)
                try:
                    symkey = self.addQuoteFromContract(c)
                except:
                    logger.warning("[{}] Live contract not found?", c)
                    continue

                extractedSymbols.add(symkey)

                # now fetch subscribed ticker
                iticker = self.quoteState.get(symkey)
                assert iticker
                # logger.info("Tracking ticker: {}", iticker)

                # record ORIGINAL symbol to ticker map so we can re-bind them after this loop
                symbolToTickerMap[symbol] = iticker

            # Replace potentially positional symbols with full symbol details we use for lookups
            predicate.symbols = frozenset(extractedSymbols)

            for extractor in predicate.extractors():
                # We prefer 'actual 'here because if this is being _repopulated_ after a reconnect(),
                # the original 'symbol' isn't valid (if it's a position request) but we added 'actual'
                # when the predicate was first created to represent the _real_ symbol data needed.
                if iticker := symbolToTickerMap.get(
                    extractor.actual or extractor.symbol
                ):
                    # Note: DO NOT .lower() 'extractor.field' because the fields are case sensitive when doing algo lookups...
                    datafield = extractor.datafield
                    timeframe = extractor.timeframe

                    # TODO: allow 'iticker' to generate text descriptions of spreads for usage in places...
                    # TODO: create helper which takes a contract and generates a compatible description we can re-parse
                    #        e.g. Future ESZ4 -> /ESZ4, options generation, index generation, ........
                    extractor.actual = self.nameForContract(iticker.contract)

                    logger.info(
                        "[{}] Assigning field extractor: {} ({}) @ {} {}",
                        pid,
                        extractor.symbol,
                        extractor.actual,
                        datafield,
                        timeframe or "",
                    )

                    assert datafield
                    fetcher = self.dataExtractorForTicker(
                        iticker, datafield, timeframe or 0
                    )

                    extractor.datafetcher = fetcher

            # now do the same for functions (if any)
            fnfetcher: (
                Callable[[dict[str, Any], str, float, float], Coroutine[Any, Any, Any]]
                | Callable[[str], Any]
            )

            for fn in predicate.functions():
                match fn.datafield.lower():
                    case "verticalput" | "vp":
                        # generate a vertical put (long) spread near requested start strike extending by point distance
                        # verticalPut(strike price for long leg, distance to short leg)
                        # Note: use negative distance to go DOWN to a lower priced short put leg
                        fnfetcher = self.ifthenExtensionVerticalSpreadPut
                    case "verticalcall" | "vc":
                        # generate a vertical put (long) spread near requested start strike extending by point distance
                        # verticalCall(strike price for long leg, distance to short leg)
                        # Note: use positive distance to go UP to a lower priced short call leg
                        fnfetcher = self.ifthenExtensionVerticalSpreadCall
                    case "position" | "pos" | "p":
                        fnfetcher = self.ifthenQuantityForContract
                    case "abs":
                        fnfetcher = self.ifthenAbs

                fn.scheduler = functools.partial(
                    self.task_create, f"[{pid}] predicate executor for {fn.datafield}"
                )
                fn.datafetcher = fnfetcher

    def dataExtractorForTicker(self, iticker: ITicker, field: str, timeframe: int):
        """Return a zero-argument function querying the live 'iticker' for 'field' and potentially 'timeframe' updates."""
        fetcher = None

        # a dot in the field means we HAVE AN ALGO! ALGO ALERT! ALGO ALERT!
        # TODO: maybe move this to an algo: prefix instead of just any dots?
        if "." in field:
            if not self.algobindertask:
                self.algobinderStart()

            assert self.algobinder

            # Note: it's up to the user ensuring a 100% correct algo field description for the full 3, 5, 8+ level depth they expect...
            return lambda *args: self.algobinder.read(field)

        def emaByField(subtype):
            parts = subtype.split(":")

            # match first component to the instance variable names of ITicker
            match parts[0]:
                case "price" | "p":
                    src = "ema"
                case "trade" | "tr":
                    src = "emaTradeRate"
                case "volume" | "vol":
                    src = "emaVolumeRate"
                case "iv":
                    src = "emaIV"
                case "delta" | "d":
                    src = "emaDelta"
                case "vega" | "v":
                    src = "emaVega"
                case _:
                    src = "ema"
                    logger.warning(
                        "No EMA source provided, defaulting to 'price' (other choices: 'delta' or 'iv' or 'vega' or 'trade' or 'volume')"
                    )

            # fetch ITicker instance variable by name
            base: TWEMA = getattr(iticker, src)

            # fetch sub-components of the EMA object
            match ":".join(parts[1:]):
                case "ema":
                    # our time-weighted ema
                    fetcher = lambda *args: base[timeframe]
                case "rms":
                    fetcher = lambda *args: base.rms()[timeframe]
                case "ema:prev:log":
                    # difference between period N and period N-1 expressed as percentage returns decayed by 'timeframe' ema
                    # (populated for all EMA durations)
                    fetcher = lambda *args: base.diffPrevLog[timeframe]
                case "ema:prev:score":
                    # CURRENT weighted sum of every period (N, N-1) pair between 0 and 6.5 hours of EMA
                    # (ranged-returns are weighted more heavily towards lower timeframes as (1/15, 1/30, 1/60, ...))
                    fetcher = lambda *args: base.diffPrevLogScore
                case "ema:prev:score:ema":
                    # EMA of weighted sum of every period (N, N-1) pair between 0 and 6.5 hours of EMA decayed by 'timeframe' ema
                    # (ranged-returns are weighted more heavily towards lower timeframes as (1/15, 1/30, 1/60, ...))
                    # Note: score:ema of [0] is the same as just the instantenous 'prev:score' too.
                    fetcher = lambda *args: base.diffPrevLogScoreEMA[timeframe]
                case "ema:vwap:log":
                    # difference between the 6.5 hour EMA and the current price expressed as percentage returns
                    # (populated for all EMA durations)
                    fetcher = lambda *args: base.diffVWAPLog[timeframe]
                case "ema:vwap:score":
                    # CURRENT weighted sum of every period (N, N-1) pair against the 6.5 hour EMA
                    fetcher = lambda *args: base.diffVWAPLogScore
                case "ema:vwap:score:ema":
                    # EMA of weighted sum of every period (N, N-1) pair against the 6.5 hour EMA decayed by 'timeframe' ema
                    fetcher = lambda *args: base.diffVWAPLogScoreEMA[timeframe]
                case _:
                    assert None, f"Invalid EMA sub-fields requested? Full request was for: {subtype}"

            return fetcher

        # We _can_ do case insenstiive matches for these:
        match field.lower():
            case "bid" | "ask":
                # Note: we use quote() bid/ask because the directly .bid/.ask values
                #       on the ticker may not be updating if IBKR breaks spread quotes
                fetcher = lambda *args: getattr(iticker.quote(), field)
            case "mid" | "midpoint" | "live":
                fetcher = lambda *args: iticker.quote().current
            case "last" | "high" | "low" | "open" | "close":
                fetcher = lambda *args: getattr(iticker.ticker, field)
            case "atr":
                # selectable as 90, 180, 300, 600, 1800 second ATRs
                fetcher = lambda *args: iticker.atrs[timeframe].atr.current
            case "sym" | "symbol":
                # just return symbol as string...
                fetcher = lambda *args: self.nameForContract(iticker.ticker.contract)  # type: ignore
            case "half":
                # half way between high and low
                fetcher = lambda *args: (iticker.ticker.high + iticker.ticker.low) / 2
            case "vwap":
                # vwap is a special case because if VWAP doesn't exist, we want to use our 6.5 hour EMA instead
                fetcher = lambda *args: iticker.ticker.vwap or iticker.ema[23_400]
            case "cost":
                # fetch live averageCost for position as reported by portfolio reporting
                # TODO: if contract is a bag with no single contract id, just add averageCost of ALL internal contract ids together?

                # Note: the IBKR portfolio API lists contracts at their multiplier-adjusted price, be want the quoted contract price,
                #       so we de-adjust them by multipliers back to contract prices again.
                # Also note: IBKR portfolio prices have positive prices but negative quantities for shorts, but we want negative prices
                #            for short positions too, so we also adjust accordingly.
                c = iticker.ticker.contract
                assert c

                contractId = c.conId
                mul = float(c.multiplier or 1)
                accountReader = self.ib.wrapper.portfolio[self.accountId]
                fetcher = lambda *args: accountReader[
                    contractId
                ].averageCost / math.copysign(accountReader[contractId].position, mul)
            case "qty":
                # fetch live qty for position as reported by portfolio reporting
                assert iticker.ticker.contract

                contractId = iticker.ticker.contract.conId
                accountReader = self.ib.wrapper.portfolio[self.accountId]
                fetcher = lambda *args: accountReader[contractId].position
            case "theta" | "delta" | "iv" | "gamma" | "d" | "g" | "t" | "v" | "vega":
                # allow some shorthand back to actual property names
                match field:
                    case "iv":
                        field = "impliedVol"
                    case "d":
                        field = "delta"
                    case "t":
                        field = "theta"
                    case "g":
                        field = "gamma"
                    case "v":
                        field = "vega"

                fetcher = lambda *args: getattr(iticker.modelGreeks, field)
            case parts if ":" in parts:
                fetcher = emaByField(parts)
            case _:
                logger.warning("Unexpected field requested? This won't work: {}", field)

        return fetcher

    @logger.catch
    def tickersUpdate(self, tickr):
        """This runs on EVERY quote update which happens 4 times per second per subsubscribed symbol.

        We don't technically need this to receive ticker updates since tickers are "live updated" in their
        own classes for reading, but we _do_ use this to calculate live metadata, reporting, or quote-based
        algo triggers (though, we could also run our own timer-based system to update once per second instead
        of running once per tick... TODO: do that instead so we run one fix-up loop maybe every 500ms to 750ms
        instead of running this callback function 200 times per second across all our symbols).

        This method should always be clean and fast because it runs up to 100+ times per second depending on how
        many tickers you are subscribed to in your client.

        Also note: because this is an ib_insync event handler, any errors or exceptions in this method are NOT
                   reported to the main program. You should attach @logger.catch to this method if you think it
                   isn't working correctly because then you can see the errors/exceptions (if any).
        """
        # logger.info("Ticker update: {}", tickr)

        for ticker in tickr:
            c = ticker.contract
            quotekey = lookupKey(c)

            try:
                # Note: we run the processTickerUpdate() before the "no bid or ask" check because some
                #       symbols like VIF/VIX/VIN and TICK-NYSE and TRIN-NYSE have 'last' values but no bid/ask on them,
                #       but we still want to process their 'last' price updates for EMA trending and alerting.
                iticker = self.quoteState[quotekey]
            except:
                # Often when we unsubscribe from a symbol, we still receive delayed ticker updates even
                # though the symbol is now deleted. Don't alert on getting updates for non-existing
                # symbols unless we need it for debugging.
                # logger.warning("Ticker update for non-existing quote: {}", quotekey)
                continue

            iticker.processTickerUpdate()

            for successCmd in self.ifthenRuntime.check(quotekey):
                match successCmd:
                    case ifthen.IfThenRuntimeSuccess(
                        pid=predicateId, cmd=cmd, predicate=p
                    ):
                        # we have a COMMAND TO RUN so SCHEDULE TO RUN A COMMAND at the next event loop wakeup
                        logger.info("Predicate Complete: {}", pp.pformat(p))
                        logger.info(
                            "[{}] Predicate scheduling command: {}", predicateId, cmd
                        )
                        self.task_create(
                            f"[{predicateId}] predicate command execution",
                            self.buildAndRun(cmd),
                        )
                    case ifthen.IfThenRuntimeError(pid=predicateId, err=e):
                        logger.warning(
                            "[{} :: [predicateId {}]] Check failed for predicate: {}",
                            quotekey,
                            predicateId,
                            str(e),
                        )

            if ticker.bid is None or ticker.ask is None:
                continue

            # only run subscription checks every 2 seconds, but run them on all symbols
            # TODO: okay, so this is now backwards. we just need a global list of things to check once per second
            #       because if we aren't triggering updates PER SYMBOL UPDATE, we don't need to track subscribers per symbol
            #       (except for stopping predicates if we remove a ticker and it has live subscribers...)
            # if time.time() - self.lastSubscriberUpdate >= 2:
            #     for iticker in self.quoteState.values():
            #         complete = set()
            #        for subscriber in iticker.subscribers:

            #   self.lastSubscriberUpdate = time.time()

            # Calculate our live metadata per update.
            # We maintain:
            #    - composite bag greeks for spreads (from each underlying leg)
            #    - EMAs for each symbol
            #    - encapsulated operations for local data details (current price vs. HOD/LOD/close percentages, etc)
            # iquote = self.iquote[quotekey]

            # this is going to be a no-op for most symbols/contracts, but when we have
            # active grabbers for this subscribed contract id, we need to run it.
            # An empty dict.get() is less than 40 ns, so we could run over 20 million of these
            # per second just for the empty dict check and it's okay.
            # The grabber check itself must also be fast and if the grabber decides it needs
            # to grab more quantity, it launches via asyncio.xreate_task() since this ticker
            # update method isn't a coroutine itself...
            # logger.info("[{}] Checking grabbers: {}", ticker.contract.conId, grabbers)
            # (this doesn't work for Bag because Bag has underlying symbols but spread prices so we can't compare "name vs. price")
            name = (c.localSymbol or c.symbol).replace(" ", "")

        # TODO: we could also run volume crossover calculations too...

        # TODO: we should also do some algo checks here based on the live quote price updates...

        # maybe also store all prices into a historical dict indexed by timestamp rounded to nearest 5, 30, 90, 300 seconds and start of day and high and low of day?
        # so we can alert on if price moving up/down by each time slot or high/low of day?
        # auto-alert on positions moving quickly
        # if ticker.contract.localSymbol in self.positions:
        #       price =
        #       if is_long:
        #           we are LONG, so we close by the BID side
        #           price = ticker.bid
        #       else:
        #          # else, we are short, so we close by the ASK side
        #           price = ticker.ask

        #   need to track previous bid/ask to determine when prices are moving for/against us

        if ICLI_DUMP_QUOTES:
            with open(
                f"tickers-{datetime.datetime.now().date()}-{self.clientId}.json", "ab"
            ) as tj:
                for ticker in tickr:
                    tj.write(
                        ourjson.dumps(
                            dict(
                                symbol=name,
                                time=str(ticker.time),
                                bid=ticker.bid,
                                bidSize=ticker.bidSize,
                                ask=ticker.ask,
                                askSize=ticker.askSize,
                                volume=ticker.volume,
                            )
                        )
                    )
                    tj.write(b"\n")

    def updateSummary(self, v):
        """Each row is populated after connection then continually
        updated via subscription while the connection remains active."""
        # logger.info("Updating sumary... {}", v)
        self.summary[v.tag] = v.value

        # regular accounts are U...; sanbox accounts are DU... (apparently)
        # Some fields are for "All" accounts under this login, which don't help us here.
        # TODO: find a place to set this once instead of checking every update?
        if self.isSandbox is None and v.account != "All":
            self.isSandbox = v.account[0] == "D"

        # TODO: we also want to maintain "Fake ITicker" for each account value so we can track it over time and use the ITicker values in ifthen statements.
        #       e.g: if :UPL > 10_000: evict *
        #       But, currently, the `ifthen` system only uses symbols and positional symbol aliases (:N) for deriving values for checking.
        # We would have to:
        #   - Create a fake/synthetic ITicker system with lookups mapping from :[AccountDetailName] or :[AccountDetailShorthand] to a fake ITicker object
        #   - Run the `.processTickerUpdate()` on the synthetic ITicker object of each new value being updated so it would trigger any `ifthen` predicate checks

        # collect updates into a single update dict so we can re-broadcast this update
        # to external agent listeners too all at once.
        update = {}
        if v.tag in STATUS_FIELDS_PROCESS:
            try:
                match v.tag:
                    case "FullMaintMarginReq":
                        update["FullMaintMarginReq"] = float(v.value)
                    case "BuyingPower":
                        # regular 25% margin for boring symbols
                        update["BuyingPower4"] = float(v.value)

                        # 30% margin for "regular" symbols
                        update["BuyingPower3"] = float(v.value) / 1.3333333333

                        # 50% margin for overnight or "really exciting" symbols
                        update["BuyingPower2"] = float(v.value) / 2
                    case "NetLiquidation":
                        update[v.tag] = float(v.value)
                    case _:
                        update[v.tag] = float(v.value)
            except:
                # don't care, just keep going
                pass
            finally:
                self.accountStatus |= update

                if v.tag == "NetLiquidation":
                    self.updateURPLPercentages()

                # TODO: resume doing this apparently
                # self.updateAgentAccountStatus("summary", update)

    def updateURPLPercentages(self):
        """Update account percentages.

        We refactored this out to its own method because there are TWO places where we need to
        calculate the URPL percentages:
          - When a new DailyPnL value is updated (during live trades once per second)
          - When a new NetLiquidation value is generated (every couple minutes)
          - When our live RealizedPnL value changes (when any trade completes an execution)

        TODO: I guess we could refactor this out to only calculate realizedpnl% on realized updates
              and unrealizedpnl% on live "dailypnl" or "unrealizedpnl" updates? Thogugh, it would
              always have to re-calculate "totalpnl%" so we only save one row of math for contiional
              single-purpose updates instead of updating all 3 of these values on each trigger.

        The "DailyPnL" stops updating after trades execute, so we need to use the "NetLiquidation" trigger
        as a fallback "flush the current result live" backup so we don't get stuck on pre-closed-trades
        percentages showing."""
        # Update {un,}realizedpnl with DailyPnL updates because DailyPnL updates is refreshed
        # once per second, while previously we were using NetLiquidation as the trigger, but
        # NetLiquidation only updates once every minute or two sometimes (so our (un)realizedpnl
        # percentages were often delayed by an annoying amount of time).
        nl = self.accountStatus.get("NetLiquidation", 1)
        upl = self.accountStatus.get("UnrealizedPnL", 0)
        rpl = self.accountStatus.get("RealizedPnL", 0)

        # Hold updates we refresh into accountStatus all at once.
        update = {}

        # Also generate some synthetic data about percentage gains we made.
        # Is this accurate enough? Should we be doing the math differently or basing it off AvailableFunds or BuyingPower instead???
        # We subtract the PnL values from the account NetLiquidation because the PnL contribution is *already* accounted for
        # in the NetLiquidation value.
        # (the updates are *here* because this runs on every NetLiq val update instead of ONLY on P&L updates)
        update["RealizedPnL%"] = (rpl / (nl - rpl)) * 100
        update["UnrealizedPnL%"] = (upl / (nl - upl)) * 100

        # Also combine realized+unrealized to show the current daily total PnL percentage because
        # maybe we have 12% realized profit but -12% unrealized and we're actually flat...
        update["TotalPnL%"] = update["RealizedPnL%"] + update["UnrealizedPnL%"]

        self.accountStatus |= update

    def updatePNL(self, v):
        """Kinda like summary, except account PNL values aren't summary events,
        they are independent PnL events. shrug.

        Also note: we merge these into our summary dict instead of maintaining
        an indepdent PnL structure.

        Also note: thse don't always get cleared automatically after a day resets,
        so if your client is open for multiple days, sometimes the previous PnL values
        still show up."""

        # TODO: keep moving average of daily PNL and trigger sounds/events
        #       if it spikes higher/lower.
        # logger.info("Updating PNL... {}", v)
        self.summary["UnrealizedPnL"] = v.unrealizedPnL
        self.summary["RealizedPnL"] = v.realizedPnL
        self.summary["DailyPnL"] = v.dailyPnL

        update = {}
        try:
            update["UnrealizedPnL"] = float(v.unrealizedPnL)
            update["RealizedPnL"] = float(v.realizedPnL)
            update["DailyPnL"] = float(v.dailyPnL)
        except:
            # don't care, just keep going
            # (maybe some of these keys don't exist yet, but they will get populated quickly as
            #  the post-connect-async-data-population finishes sending us data for all the fields)
            pass
        finally:
            self.accountStatus |= update
            self.updateURPLPercentages()
            # ignore agent pnl update for now since it is probably in the summary updates anyway?
            # self.updateAgentAccountStatus("pnl", update)

    def updatePNLSingle(self, v):
        """Streaming individual position PnL updates.

        Must be requested per-position.

        The reqPnLSingle method is the only way to get
        live 'dailyPnL' updates per position (updated once per second!)."""

        # logger.info("Updating PNL... {}", v)
        # These are kept "live updated" too, so just save the
        # return value after the subscription.
        self.pnlSingle[v.conId] = v

    def updateAgentAccountStatus(
        self, category: str, update: FillReport | dict[str, float | int]
    ):
        """Update internal account data (and maybe external accounting) when trade details get updated.

        Send the `update` dict to current agent server so agent can know about our portfolio for making decisions.

        Note: this method isn't async because it's called from the ib_insync update callbacks, which themselves aren't async,
              but we can just create tasks for updating instead.
        """

        match category:
            case "commission":
                # New trade event occurred, so let's record it in our local trade-order-position-quantity-stop tracker.
                assert isinstance(update, FillReport)

                # TODO: we also need a "generic sync" mechanism to update _current_ positions into the state if we don't have
                #       executions against them. Basically: list all positions, and if a contractId isn't in our ordermgr, just
                #       add it. We also need additional ordermgr management tools for gluing and ungluing positions possibly.
                self.ordermgr.add_trade(
                    # using just raw contract id as the identifier for now... we can look it up every time we want it resolved I guess.
                    update.conId,
                    OrderMgrTrade(
                        # We scope order ids _per client_ since IBKR request IDs are only per-client.
                        # This also assumes you never "reset request IDs" in your gateway, but this
                        # feature is mainly for tracking trades occurring near in time together.
                        orderid=(self.clientId, update.orderId),
                        price=update.price,
                        qty=update.qty,
                        timestamp=update.when,
                        commission=update.commission,
                    ),
                )
            case "summary":
                pass

        if False:
            if self.agent:

                async def sendUpdate():
                    if not isinstance(update, dict):
                        send = asdict(update)
                        await self.agent.sendAccountUpdate(
                            ourjson.dumps({category: send})
                        )

                self.task_create("Send Account Update Payload", sendUpdate())

    def bottomToolbar(self):
        self.updates += 1
        self.updatesReconnect += 1
        self.now = whenever.ZonedDateTime.now("US/Eastern")
        self.nowpy = self.now.py_datetime()

        def fmtPrice2(n: float):
            # Some prices may not be populated if they haven't
            # happened yet (e.g. PNL values if no trades for the day yet, etc)
            if not n:
                n = 0

            # if GTE $1 million, stop showing cents.
            if n > 999_999.99:
                return f"{n:>10,.0f}"

            return f"{n:>10,.2f}"

        def fmtEquitySpread(n, digits=2):
            if isinstance(n, (int, float)):
                if n < 1000:
                    return f"{n:>6.{digits}f}"

                return f"{n:>6,.0f}"

            return f"{n:>5}"

        def fmtPriceOpt(n, digits=2):
            return f"{n or nan:>5,.{digits}f}"

        # Fields described at:
        # https://ib-insync.readthedocs.io/api.html#module-ib_insync.ticker

        useLast = self.localvars.get("last")
        hideSingleLegs = self.localvars.get("hide")
        hideMissing = self.localvars.get("hidemissing")

        def formatTicker(c):
            ls = lookupKey(c.contract)

            # ibkr API keeps '.close' as the previous full market day close until the next
            # full market day, so for example over the weekend where there isn't a new "full
            # market day," the '.close' is always Thursday's close, while '.last' will be the last
            # traded value seen, equal to Friday's last after-hours trade.
            # But when a new market day starts (but before trading begins), the 'c.last' becomes
            # nan and '.close' becomes the actual expected "previous market day" close we want
            # to use.
            # In summary: '.last' is always the most recent traded price unless it's a new market
            # day before market open, then '.last' is nan and '.close' is the previous most accurate
            # (official) close price, but doesn't count AH trades (we think).
            # Also, this price is assuming the last reported trade is accurate to the current
            # NBBO spread because we aren't checking "if last is outside of NBBO, use NBBO midpoint
            # instead" because these are for rather active equity symbols (we do use the current
            # quote midpoint as price for option pricing though due to faster quote-vs-trade movement)

            # We switched from using "lastPrice" as the shown price to the current midpoint
            # as the shown price because sometimes we were getting price lags when midpoints
            # shifted faster than buying or selling, so we were looking at outdated "prices"
            # for some decisions.
            match c.quote():
                case (
                    QuoteSizes(
                        bid=bid,
                        ask=ask,
                        bidSize=bidSize,
                        askSize=askSize,
                        last=last,
                        close=close,
                    ) as qs
                ):
                    usePrice = qs.last if useLast else qs.current

            high = c.high
            low = c.low
            vwap = c.vwap
            decimals: int | None

            # short circuit the common case of simple quotes
            if isinstance(c.contract, (Stock, Option, Bag)):
                # NOTE: this can potentially miss the case of Bags of FuturesOption having higher precision, but not a priority at the moment.
                decimals = 2
            else:
                try:
                    # NOTE: decimals *can* be zero, so our decimal fetcher returns None on failure to load, so None means "wait for data to populate"
                    if (decimals := self.idb.decimals(c.contract)) is None:
                        return f"WAITING TO POPULATE METADATA FOR: {c.contract.localSymbol}"

                    # for DISPLAY purposes, don't allow one digit decimals (things like /RTY trade in $0.1 increments, but we still want to show $0.10 values)
                    # NOTE: don't make this min(2, decimals) because we _DO_ want to allow 0 decimals, but deny only 1 decimals.
                    if decimals == 1:
                        decimals = 2

                except Exception as e:
                    # logger.exception("WHY?")
                    return f"METADATA LOOKUP FAILED {e}, WAITING TO TRY AGAIN FOR: {c.contract.localSymbol}"

            # assert decimals >= 0, f"Why bad decimals here for {c.contract}?"

            if (bid is None and ask is None) and (usePrice is None):
                name = c.contract.localSymbol
                if isinstance(c.contract, Bag):
                    try:
                        name = " :: ".join(
                            [
                                f"{z.action:<5} {z.ratio:>3} {self.conIdCache.get(z.conId).localSymbol.replace(' ', ''):>20}"
                                for z in c.contract.comboLegs
                            ]
                        )
                    except:
                        # just give it a try. if it doesn't work, no problem.
                        pass

                if hideMissing:
                    return None

                return f"WAITING FOR LIVE MARKET DATA: {name:>12}  ::  {bid=} x {bidSize=}  {ask=} x {askSize=}  {last=} {close=} {usePrice=} {high=} {low=}"

            if usePrice is None:
                if hideMissing:
                    return None

                return f"WAITING FOR DATA UPDATE: {c.contract.localSymbol}"

            if c.lastTimestamp:
                agoLastTrade = as_duration(
                    (self.nowpy - c.lastTimestamp).total_seconds()
                )
            else:
                agoLastTrade = None

            if c.time:
                ago = as_duration((self.nowpy - c.time).total_seconds())
            else:
                ago = "NO LIVE DATA"

                # since no live data, use our synthetic midpoint to update quote history for now
                # (This should only apply to spreads because if a single leg or single stock has no quote data, we have no data at all...)
                # (for some reason, sometimes IBKR just refuses to quote spreads even though all the legs have live offerings)
                # quotekey = lookupKey(c.contract)
                # self.quotehistory[quotekey].append(usePrice)

            percentVWAP, amtVWAP = c.percentAmtFromVWAP()
            percentUnderHigh, amtHigh = c.percentAmtFromHigh()
            percentUpFromLow, amtLow = c.percentAmtFromLow()
            percentUpFromClose, amtClose = c.percentAmtFromClose()

            # If there are > 1,000 point swings, stop displaying cents.
            # also the point differences use the same colors as the percent differences
            # because having fixed point color offsets doesn't make sense (e.g. AAPL moves $2
            # vs DIA moving $200)

            # if bidsize or asksize are > 100,000, just show "100k" instead of breaking
            # the interface for being too wide
            if not bidSize:
                b_s = f"{'X':>6}"
            elif 0 < bidSize < 1:
                # there's a bug here when 'bidSize' is 'inf' and it's triggering here??
                b_s = f"{bidSize:>6.4f}"
            elif bidSize < 100_000:
                b_s = f"{int(bidSize):>6,}"
            else:
                b_s = f"{int(bidSize // 1000):>5,}k"

            if not askSize:
                a_s = f"{'X':>6}"
            elif 0 < askSize < 1:
                a_s = f"{askSize:>6.4f}"
            elif askSize < 100_000 or (askSize != askSize):
                a_s = f"{int(askSize):>6,}"
            else:
                a_s = f"{int(askSize // 1000):>5,}k"

            # use different print logic if this is an option contract or spread
            if isinstance(c.contract, (Option, FuturesOption, Bag)):
                # if c.modelGreeks:
                #     mark = c.modelGreeks.optPrice

                mark = round((bid + ask) / 2, decimals) if bid and ask else 0

                e100 = round(c.ema[60], decimals)
                e300 = round(c.ema[300], decimals)

                # logger.info("[{}] Got EMA for OPT: {} -> {}", ls, e100, e300)
                e100diff = (mark - e100) if e100 else 0

                ediff = e100 - e300
                if ediff > 0:
                    trend = "&gt;"
                elif ediff < 0:
                    trend = "&lt;"
                else:
                    trend = "="

                # For options, instead of using percent difference between
                # prices, we use percent return over the low/close instead.
                # e.g. if low is 0.05 and current is 0.50, we want to report
                #      a 900% multiple, not a 163% difference between the
                #      two numbers as we would report for normal stock price changes.
                # Also note: we use 'mark' here because after hours, IBKR reports
                # the previous day open price as the current price, which clearly
                # isn't correct since it ignores the entire most recent day.
                bighigh = (((mark / high) - 1) * 100) if high else None

                # only report low if current mark estimate is ABOVE the registered
                # low for the day, else we report it as currently trading AT the low
                # for the day instead of potentially BELOW the low for the day.
                biglow = (((mark / low) - 1) * 100) if low else None
                bigclose = (((mark / close) - 1) * 100) if close else None

                emptyFieldA = "       "
                emptyFieldB = "        "
                und = nan
                underlyingStrikeDifference = None
                iv = None
                delta = None
                theta = None
                try:
                    iv = c.modelGreeks.impliedVol
                    delta = c.modelGreeks.delta
                    theta = c.modelGreeks.theta

                    # Note: keep underlyingStrikeDifference the LAST attempt here because if the user doesn't
                    #       have live market data for this option, then 'und' is 0 and this math breaks,
                    #       but if it breaks _last_ then the greeks above still work properly.
                    strike = c.contract.strike
                    und = c.modelGreeks.undPrice
                    underlyingStrikeDifference = -(strike - und) / und * 100
                except:
                    pass

                # Note: we omit OPEN price because IBKR doesn't report it (for some reason?)
                # greeks available as .bidGreeks, .askGreeks, .lastGreeks, .modelGreeks each as an OptionComputation named tuple.
                rowName: str

                # Note: we generate a manual legsAdjust to de-adjust our width measurements if this is
                #       an (assumed) equal width IC spread. (e.g. 2 legs are 1 width, but 4 legs are the width of only *1 pair*)
                legsAdjust: float = 1.0

                # For all combos, we cache the ID to original symbol mapping
                # after the contractId is resolved.
                if c.contract.comboLegs:
                    legsAdjust = len(c.contract.comboLegs) / 2

                    # generate rows to look like:
                    # B  1 AAPL 212121 C 000...
                    # S  2 ....
                    rns = []

                    # for puts, we want legs listed HIGH to LOW
                    # for calls, we want legs listed LOW to HIGH
                    for idx, x in enumerate(
                        sorted(
                            c.contract.comboLegs,
                            # TODO: move this to a leg-sort-lookup cache so it's less work to run every time
                            key=lambda leg: sortLeg(leg, self.conIdCache),
                        )
                    ):
                        try:
                            contract = self.conIdCache[x.conId]
                        except:
                            # cache is broken for this contract id...
                            return f"[CACHE BROKEN FOR {x=} in {c.contract=}]"

                        padding = "    " if idx > 0 else ""
                        action = " "
                        right = " "

                        # localSymbol is either:
                        #  - AAPL  240816C00220000 (contract leg)
                        #  - AAPL (stock leg)

                        # We want to split out some of the details with spaces if it's a full option symbol because
                        # we show spreads as vertically stacked and it makes it easier to pick out which legs are long
                        # vs short vs their strikes and dates as compared to the default OCC showing we use elsewhere.
                        name = contract.localSymbol
                        # TODO: turn this into a top-level @cache function? It just reformats the contract output every time.
                        if isinstance(contract, (Option, Future, FuturesOption)):
                            date = contract.lastTradeDateOrContractMonth
                            right = contract.right or "U"
                            strike = contract.strike
                            action = x.action[0]

                            # TODO: fix padding length between HAS digits (longer) and NOT HAS digits (shorter)
                            #       like if strikes are 225.50 and 230.
                            # Need to: detect decimals for _all_ strikes then do max(legdigits)
                            legstrikedigits = 0 if int(strike) == strike else 2

                            # highlight BUY strikes as bold so we can easily pick them out of spreads
                            # https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/styling.html
                            if strike:
                                if action == "B":
                                    strikeFormatted = f"<aaa bg='ansibrightblue'>{strike:>5,.{legstrikedigits}f}</aaa>"
                                else:
                                    strikeFormatted = f"{strike:>5,.{legstrikedigits}f}"
                            else:
                                strikeFormatted = ""

                            name = (
                                f"{contract.symbol:<4} {date} {right} {strikeFormatted}"
                            )

                        rns.append(f"{padding}{action} {right} {x.ratio:2} {name}")

                    rowName = "\n".join(rns)

                    if False:
                        logger.info(
                            "Contract and vals for combo: {}  -> {} -> {} -> {} -> {}",
                            c.contract,
                            ls,
                            e100,
                            e300,
                            (usePrice, c.bid, c.ask, c.high, c.low),
                        )

                    # show a recent range of prices since spreads have twice (or more) the bid/ask volatility
                    # of a single leg option (due to all the legs being combined into one quote dynamically)
                    src = c.history

                    # typically, a low stddev indicates temporary low volatility which is
                    # the calm before the storm when a big move happens next (in either direction,
                    # but direction prediction can be augmented with moving average crossovers).
                    try:
                        std = statistics.stdev(src)
                    except:
                        std = 0

                    try:
                        parts: list[str | float] = [
                            round(x, 2)
                            for x in statistics.quantiles(
                                src,
                                n=5,
                                method="inclusive",
                            )
                        ]

                        # TODO: benchmark if this double min/max run is faster than looping over it once and just
                        #       checking for min/max in one loop instead of two loops here.
                        minmax = max(src) - min(src)
                    except:
                        # 'statistics' throws an exception if there's not enough data points yet...
                        parts = sorted(src)
                        minmax = 0

                    # add marker where curent price goes in this range...
                    # (sometimes this complains for some reason, but it clears up eventually)
                    try:
                        bpos = bisect.bisect_left(parts, mark)
                        parts.insert(bpos, "[X]")
                    except:
                        logger.info("Failed parts on: {}", parts)
                        pass

                    partsFormatted = ", ".join(
                        [
                            f"{x:>7.2f}" if isinstance(x, (float, int)) else x
                            for x in parts
                        ]
                    )

                    bighigh, amtHigh = c.percentAmtFromHigh()
                    biglow, amtLow = c.percentAmtFromLow()
                    percentCollectiveVWAP, amtCollectiveVwap = c.percentAmtFromVWAP()

                    pctBigHigh, amtBigHigh = (
                        mkPctColor(
                            bighigh,
                            [
                                f"{bighigh:>7.2f}%"
                                if bighigh < 10_000
                                else f"{bighigh:>7,.0f}%",
                                f"{amtHigh:>7.2f}"
                                if amtHigh < 1000
                                else f"{amtHigh:>7,.0f}",
                            ],
                        )
                        if bighigh is not None
                        else (emptyFieldA, emptyFieldB)
                    )
                    pctBigLow, amtBigLow = (
                        mkPctColor(
                            biglow,
                            [
                                f"{biglow:>7.2f}%"
                                if biglow < 10_000
                                else f"{biglow:>7,.0f}%",
                                f"{amtLow:>7.2f}"
                                if amtLow < 1000
                                else f"{amtLow:>7,.0f}",
                            ],
                        )
                        if biglow is not None
                        else (emptyFieldA, emptyFieldB)
                    )
                    _pctBigVWAP, amtBigVWAPColor = (
                        mkPctColor(
                            percentCollectiveVWAP,
                            ["", f"{amtCollectiveVwap:>6.{decimals}f}"],
                        )
                        if percentCollectiveVWAP is not None
                        else (
                            "      ",
                            "      ",
                        )
                    )

                    # Some of the daily values seem to exist for spreads: high and low of day, but previous day close just reports the current price.
                    # this is OPTION BAG/SPREAD ROWS
                    # fmt: off
                    g = c.ticker.modelGreeks
                    if g:
                        collectiveIV = g.impliedVol
                        collectiveDelta = g.delta
                        collectiveTheta = g.theta
                        collectiveVega = g.vega
                        collectiveWidth = c.width
                        wpts = c.width - mark
                        if (
                            collectiveWidth == collectiveWidth
                            and int(collectiveWidth) == collectiveWidth
                        ):
                            widthCents = 0
                        else:
                            widthCents = 2
                    else:
                        collectiveIV = nan
                        collectiveDelta = nan
                        collectiveTheta = nan
                        collectiveVega = nan
                        collectiveWidth = nan
                        widthCents = 0
                        wpts = nan

                    return " ".join(
                        [
                            rowName,
                            f"[iv {collectiveIV or 0:>5.2f}]",
                            f"[d {collectiveDelta or 0:>5.2f}]",
                            f"[t {collectiveTheta or 0:>6.2f}]",
                            f"[v {collectiveVega or 0:>5.2f}]",
                            f"[w {collectiveWidth or 0:>3.{widthCents}f}]",
                            f"{fmtPriceOpt(e100):>6}",
                            f"{trend}",
                            f"{fmtPriceOpt(e300):>6}",
                            f" {fmtPriceOpt(mark):>5} ±{fmtPriceOpt((ask or nan) - mark, decimals):<4}",
                            f" ({pctBigHigh} {amtBigHigh} {fmtPriceOpt(high):>6})" if high else " (                       )",
                            f"({pctBigLow} {amtBigLow} {fmtPriceOpt(low):>6})" if low else "(                       )",
                            f" {fmtPriceOpt(bid):>6} x {b_s}   {fmtPriceOpt(ask):>6} x {a_s}",
                            f"{amtBigVWAPColor}",
                            f" ({ago:>7})",
                            f"  :: {partsFormatted}  (r {minmax:.2f}) (s {std:.2f}) (w {wpts / legsAdjust:.2f}; {collectiveWidth / (mark or 1) / legsAdjust:,.1f}x)",
                            "HALTED!" if c.halted else "",
                        ]
                    )
                    # fmt: on
                else:
                    if hideSingleLegs:
                        return None

                    if isinstance(c.contract, FuturesOption):
                        strike = c.contract.strike
                        if strike == (istrike := int(c.contract.strike)):
                            strike = istrike

                        fparts = c.contract.localSymbol.split()
                        tradingClass = fparts[0][:-2]
                        month = fparts[0][3:]
                        expiration = fparts[1][1:]
                        rowBody = f"{tradingClass} {month} {expiration} {c.contract.right} {c.contract.lastTradeDateOrContractMonth[2:]} {strike}"
                        rowName = f"{rowBody:<21}:"
                    else:
                        rowName = f"{c.contract.localSymbol:<21}:"

                    # color spreads using our CUSTOM synthetic high/low indicators
                    pctBigHigh, amtBigHigh = (
                        mkPctColor(
                            bighigh,
                            [
                                f"{bighigh:>7.2f}%"
                                if bighigh < 10_000
                                else f"{bighigh:>7,.0f}%",
                                f"{amtHigh:>7.2f}"
                                if amtHigh < 1000
                                else f"{amtHigh:>7,.0f}",
                            ],
                        )
                        if bighigh is not None
                        else (emptyFieldA, emptyFieldB)
                    )
                    pctBigLow, amtBigLow = (
                        mkPctColor(
                            biglow,
                            [
                                f"{biglow:>7.2f}%"
                                if biglow < 10_000
                                else f"{biglow:>7,.0f}%",
                                f"{amtLow:>7.2f}"
                                if amtLow < 1000
                                else f"{amtLow:>7,.0f}",
                            ],
                        )
                        if biglow is not None
                        else (emptyFieldA, emptyFieldB)
                    )
                    pctBigClose, amtBigClose = (
                        mkPctColor(
                            bigclose,
                            [
                                f"{bigclose:>7.2f}%"
                                if bigclose < 1000
                                else f"{bigclose:>7,.0f}%",
                                f"{amtClose:>7.2f}"
                                if amtClose < 10_000
                                else f"{amtClose:>7,.0f}",
                            ],
                        )
                        if bigclose is not None
                        else (emptyFieldA, emptyFieldB)
                    )

                    if isinstance(c.contract, (Option, FuturesOption)):
                        # has data like:
                        # FuturesOption(conId=653770578, symbol='RTY', lastTradeDateOrContractMonth='20231117', strike=1775.0, right='P', multiplier='50', exchange='CME', currency='USD', localSymbol='R3EX3 P1775', tradingClass='R3E')
                        ltdocm = c.contract.lastTradeDateOrContractMonth
                        y = ltdocm[2:4]
                        m = ltdocm[4:6]
                        d = ltdocm[6:8]
                        pc = c.contract.right
                        price = c.contract.strike
                        # sym = rowName

                    # Note: this dynamic calendar math shows the exact time remaining even accounting for (pre-scheduled) early market close days.
                    when = (
                        fetchEndOfMarketDayAtDate(2000 + int(y), int(m), int(d))
                        - self.now
                    ).in_days_of_24h()

                    # this may be too wide for some people? works for me.
                    # just keep shrinking your terminal font size until everything fits?
                    # currently works nicely via:
                    #   - font: Monaco
                    #   - size: 10
                    #   - terminal width: 275+ characters
                    #   - terminal height: 60+ characters

                    # guard the ITM flag because after hours 'underlying price' isn't populated in option quotes
                    itm = ""
                    if delta and und and mark:
                        if delta > 0 and und >= price:
                            # calls
                            itm = "I"
                        elif delta < 0 and und <= price:
                            # puts
                            itm = "I"

                    # "compensated" is acquisiton price for the underlying if you short this strike.
                    # basically (strike - premium) == price of underlying if you get assigned.
                    # (here, "price"  is the "strike price" in the contract)
                    # provide defaults due to async value population from IBKR (and sometimes we don't have underlying price if we don't have market data)
                    compdiff = 0.0
                    try:
                        match pc:
                            case "P":
                                # for puts, we calculate break-even short prices BELOW the the underlying.
                                # first calculate the premium difference from the strike price,
                                compensated = price - mark
                                # then calculate how far from the underlying for the break-even-at-expiry price.
                                # (here, underlying is ABOV E the (strike - premium break-even))
                                compdiff = und - compensated
                            case "C":
                                # for calls, we calculate break-even short prices ABOVE the the underlying.
                                compensated = price + mark
                                # same as above, but for shorting calls, your break-even is above the underlying.
                                # (here, underlying is BELOW the (strike + premium break-even))
                                compdiff = compensated - und
                    except:
                        pass

                    # signal if the current option midpoint is higher or lower than the IBKR theoretical value
                    # (we _can_ do this, but not sure it's really useful to show)
                    # modelDiff = " "
                    # try:
                    #     modelDiff = "+" if round(c.modelGreeks.optPrice, 2) > mark else "-"
                    # except:
                    #     # ignore model not existing when quotes are first added
                    #     pass

                    _pctVWAP, amtVWAPColor = (
                        mkPctColor(
                            percentVWAP,
                            ["", f"{amtVWAP:>6.{decimals}f}"],
                        )
                        if amtVWAP is not None
                        else (
                            "      ",
                            "      ",
                        )
                    )

                    # this is SINGLE LEG OPTION ROWS
                    # fmt: off
                    return " ".join(
                        [
                            rowName,
                            f"[u {und or np.nan:>8,.2f} ({itm:<1} {underlyingStrikeDifference or np.nan:>7,.2f}%)]",
                            f"[iv {iv or np.nan:.2f}]",
                            f"[d {delta or np.nan:>5.2f}]",
                            # do we want to show theta or not? Not useful for intra-day trading and we have it in `info` output anyway too.
                            # f"[t {theta or np.nan:>5.2f}]",
                            f"{fmtPriceOpt(e100):>6}",
                            f"{trend}",
                            f"{fmtPriceOpt(e300):>6}",
                            f"{fmtPriceOpt(mark or (c.modelGreeks.optPrice if c.modelGreeks else 0)):>6} ±{fmtPriceOpt((ask or np.nan) - mark, decimals):<4}",
                            f"({pctBigHigh} {amtBigHigh} {fmtPriceOpt(high):>6})" if high else "(                       )",
                            f"({pctBigLow} {amtBigLow} {fmtPriceOpt(low):>6})" if low else "(                       )",
                            f"({pctBigClose} {amtBigClose} {fmtPriceOpt(close):>6})" if close else "(                       )",
                            f" {fmtPriceOpt(bid or np.nan):>6} x {b_s}   {fmtPriceOpt(ask or np.nan):>6} x {a_s}",
                            f"{amtVWAPColor}",
                            f" ({ago:>7})",
                            f" (s {compensated:>8,.2f} @ {compdiff:>6,.2f})",
                            f" ({when:>3.2f} d)" if when >= 1 else f" ({as_duration(when * 86400)})",
                            "HALTED!" if c.halted else "",
                        ]
                    )
                    # fmt: ond

            # TODO: pre-market and after-market hours don't update the high/low values, so these are
            #       not populated during those sessions.
            #       this also means during after-hours session, the high and low are fixed to what they
            #       were during RTH and are no longer valid. Should this have a time check too?
            pctVWAP, amtVWAPColor = (
                mkPctColor(
                    percentVWAP,
                    [
                        f"{percentVWAP:>6.2f}%",
                        f"{amtVWAP:>8.{decimals}f}"
                        if amtVWAP < 1000
                        else f"{amtVWAP:>8.0f}",
                    ],
                )
                if amtVWAP is not None
                else (
                    "       ",
                    "        ",
                )
            )

            pctUndHigh, amtUndHigh = (
                mkPctColor(
                    percentUnderHigh,
                    [
                        f"{percentUnderHigh:>6.2f}%",
                        f"{amtHigh:>8.{decimals}f}"
                        if amtHigh < 1000
                        else f"{amtHigh:>8.0f}",
                    ],
                )
                if amtHigh is not None
                else ("       ", "        ")
            )

            pctUpLow, amtUpLow = (
                mkPctColor(
                    percentUpFromLow,
                    [
                        f"{percentUpFromLow:>5.2f}%",
                        f"{amtLow:>6.{decimals}f}"
                        if amtLow < 1000
                        else f"{amtLow:>6.0f}",
                    ],
                )
                if amtLow is not None
                else ("      ", "      ")
            )

            # high and low are only populated after regular market hours, so allow nan to show the
            # full float value during pre-market hours.
            pctUpClose, amtUpClose = (
                mkPctColor(
                    percentUpFromClose,
                    [
                        f"{percentUpFromClose:>6.2f}%",
                        f"{amtClose:>8.{decimals}f}"
                        if amtClose < 1000
                        else f"{amtClose:>8.0f}",
                    ],
                )
                if amtClose is not None
                else ("      ", "         ")
            )

            # somewhat circuitous logic to format NaNs and values properly at the same string padding offsets
            # Showing the 3 minute ATR by default. We have other ATRs to choose from. See per-symbol 'info' output for all live values.
            atrval = c.atrs[180].atr.current

            # if ATR > 100, omit cents so it fits in the narrow column easier
            if atrval > 100:
                atr = f"{atrval:>5.0f}"
            else:
                # else, we can print a full width value since it will fit in the 5 character width column
                atr = f"{atrval:>5.2f}"

            e100 = round(c.ema[60], decimals)
            e300 = round(c.ema[300], decimals)

            # for price differences we show the difference as if holding a LONG position
            # at the historical price as compared against the current price.
            # (so, if e100 is $50 but current price is $55, our difference is +5 because
            #      we'd have a +5 profit if held from the historical price.
            #      This helps align "price think" instead of showing difference from historical
            #      vs. current where "smaller historical vs. larger current" would cause negative
            #      difference which is actually a profit if it were LONG'd in the past)
            # also don't show differences for TICK because it's not really a useful number (and it's too big breaking formatting)
            if ls == "TICK-NYSE":
                e100diff = 0
                e300diff = 0
            else:
                e100diff = (usePrice - e100) if e100 else 0
                e300diff = (usePrice - e300) if e300 else 0
            # logger.info("[{}] e100 e300: {} {} {} {}", ls, e100, e300, e100diff, e300diff)

            # also add a marker for if the short term trend (1m) is GT, LT, or EQ to the longer term trend (3m)
            ediff = e100 - e300
            if ediff > 0:
                trend = "&gt;"
            elif ediff < 0:
                trend = "&lt;"
            else:
                trend = "="

            # fmt: off
            return " ".join(
                [
                    f"{ls:<9}",
                    f"{e100:>10,.{decimals}f}",
                    f"({e100diff:>6,.2f})" if e100diff else "(      )",
                    f"{trend}",
                    f"{e300:>10,.{decimals}f}",
                    f"({e300diff:>6,.2f})" if e300diff else "(      )",
                    f"{usePrice:>10,.{decimals}f} ±{fmtEquitySpread(ask - usePrice, decimals) if (ask and ask >= usePrice) else '':<6}",
                    f"({pctUndHigh} {amtUndHigh})",
                    f"({pctUpLow} {amtUpLow})",
                    f"({pctUpClose} {amtUpClose})",
                    f"{high or np.nan:>10,.{decimals}f}",
                    f"{low or np.nan:>10,.{decimals}f}",
                    f"<aaa bg='purple'>{c.bid or np.nan:>10,.{decimals}f} x {b_s} {ask or np.nan:>10,.{decimals}f} x {a_s}</aaa>",
                    f"({atr})",
                    f"({pctVWAP} {amtVWAPColor})",
                    f"{close or np.nan:>10,.{decimals}f}",
                    f"({ago:>7})",
                    # Only show "last trade ago" if it is recent enough
                    f"@ ({agoLastTrade})" if agoLastTrade else "",
                    "     HALTED!" if c.halted else "",
                ]
            )
            # fmt: on

        try:
            rowlen, _ = shutil.get_terminal_size()

            rowvals: list[list[str]] = [[]]
            currentrowlen = 0
            DT = []
            for cat, val in self.accountStatus.items():
                # if val == 0:
                #    continue

                # Note: if your NLV is >= $25,000 USD, then the entire
                #       DayTradesRemaining{,T+{1,2,3,4}} sections do not
                #       show up in self.accountStatus anymore.
                #       This also means if you are on the border of $25k ± 0.01,
                #       the field will keep vanishing and showing up as your
                #       account values bounces above and below the PDT threshold
                if cat.startswith("DayTrades"):
                    # the only field we treat as just an integer

                    # skip field if is -1, meaning account is > $25k so
                    # there is no day trade restriction
                    if val == -1:
                        continue

                    DT.append(int(val))

                    # wait until we accumulate all 5 day trade indicators
                    # before printing the day trades remaining count...
                    if len(DT) < 5:
                        continue

                    section = "DayTradesRemaining"
                    # If ALL future day trade values are equal, only print the
                    # single value.
                    if all(x == DT[0] for x in DT):
                        value = f"{section:<20} {DT[0]:>14}"
                    else:
                        # else, there is future day trade divergence,
                        # so print all the days.
                        csv = ", ".join([str(x) for x in DT])
                        value = f"{section:<20} ({csv:>14})"
                else:
                    # else, use our nice formatting
                    # using length 14 to support values up to 999,999,999.99
                    value = f"{cat:<20} {fmtPrice2(val):>14}"

                vlen = len(value)
                # "+ 4" because of the "    " in the row entry join
                # ALSO, limit each row to 7 elements MAX so we always have the same status block
                # alignment regardless of console width (well, if consoles are wide enough for six or seven columns
                # at least; if your terminal is smaller than six status columns the entire UI is probably truncated anyway).
                totLen = currentrowlen + vlen + 4
                if (totLen < rowlen) and (totLen < 271):
                    # append to current row
                    rowvals[-1].append(value)
                    currentrowlen += vlen + 4
                else:
                    # add new row, reset row length
                    rowvals.append([value])
                    currentrowlen = vlen

            balrows = "\n".join(["    ".join(x) for x in rowvals])

            # RegT overnight margin means your current margin balance must be less than your SMA value.
            # Your SMA account increases with deposits and when your positions grow profit, so the minimum
            # overnight you can hold is 50% of your deposited cash, while the maximum you can hold is your
            # 4x margin if your SMA has grown larger than your total BuyingPower.
            # After you trade for a while without withdraws, your profits will grow your SMA value to be larger
            # than your full 4x BuyingPower, so eventually you can hold 4x margin overnight with no liquidations.
            # (note: the SMA margin calculations are only for RegT and do not apply to portfolio margin / SPAN accounts)
            overnightDeficit = self.accountStatus["SMA"]

            onc = ""
            if overnightDeficit < 0:
                # You must restore your SMA balance to be positive before:
                # > Whenever you have a position change on a trading day,
                # > we check the balance of your SMA at the end of the US trading day (15:50-17:20 ET),
                # > to ensure that it is greater than or equal to zero.
                onc = f" (OVERNIGHT REG-T MARGIN CALL: ${-overnightDeficit:,.2f})"

            # some positions have less day margin than overnight margin, and we can see the difference
            # where 'FullMaintMarginReq' is what is required after RTH closes and 'MaintMarginReq' is required for the current session.
            # Just add a visible note if our margin requirements will increase if we don't close out live positions.
            fmm = self.accountStatus.get("FullMaintMarginReq", 0)
            mm = self.accountStatus["MaintMarginReq"]

            if fmm > mm:
                onc += f" (OVERNIGHT MARGIN LARGER THAN DAY: ${fmm:,.2f} (+${fmm - mm:,.2f}))"

            qs = self.quoteStateSorted

            spxbreakers = ""

            try:
                spx = self.quoteState.get("SPX")
                if spx:
                    # hack around IBKR quotes being broken over weekends/holdays
                    # NOTE: this isn't valid across weekends because until Monday morning, the "close" is "Thursday close" not frday close. sigh.
                    #       also the SPX symbol never has '.open' value so we can't detect "stale vs. current quote from last close"
                    spxl = spx.last
                    spxc = spx.close

                    def undX(spxd, spxIn):
                        return (spxd / spxIn) * 100

                    spxc7 = round(spxc / 1.07, 2)
                    spxcd7 = round(spxl - spxc7, 2)

                    spxc13 = round(spxc / 1.13, 2)
                    spxcd13 = round(spxl - spxc13, 2)

                    spxc20 = round(spxc / 1.20, 2)
                    spxcd20 = round(spxl - spxc20, 2)

                    spxbreakers = "   ".join(
                        [
                            f"7%: {spxc7:,.2f} ({spxcd7:,.2f}; {undX(spxcd7, spxc7):.2f}%)",
                            f"13%: {spxc13:,.2f} ({spxcd13:,.2f}; {undX(spxcd13, spxc13):.2f}%)",
                            f"20%: {spxc20:,.2f} ({spxcd20:,.2f}; {undX(spxcd20, spxc20):.2f}%)",
                        ]
                    )
            except:
                # the data will populate eventually
                # logger.exception("cant update spx?")
                pass

            # TODO: we may want to iterate these to exclude "Inactive" or orders like:
            # [x.log[-1].status == "Inactive" for x in self.ib.openTrades()]
            # We could also exclude waiting bracket orders when status == 'PreSubmitted' _and_ has parentId
            ordcount = len(self.ib.openTrades())
            openorders = f"open orders: {ordcount:,}"

            positioncount = len(self.ib.portfolio())
            openpositions = f"positions: {positioncount:,}"

            executioncount = len(self.ib.fills())
            todayexecutions = f"executions: {executioncount:,}"

            # TODO: We couold also flip this between a "time until market open" vs "time until close" value depending
            #       on if we are out of market hours or not, but we aren't bothering with the extra logic for now.
            untilClose = (
                fetchEndOfMarketDayAtDate(self.now.year, self.now.month, self.now.day)
                - self.now
            )
            todayclose = f"mktclose: {convert_time(untilClose.in_seconds())}"
            daysInMonth = f"dim: {tradingDaysRemainingInMonth()}"
            daysInYear = f"diy: {tradingDaysRemainingInYear()}"

            # this weird thing lets is optionally remove tickers by letting formatTicker() return None, then we drop None results from showing.
            rows = []

            # this extra processing loop for format inclusion lets us _optionally hide_ ticker rows
            # from appearing in the toolbar numbered list.
            # If you enable icli env var 'hide', currently all single-leg option rows get removed from
            # printing (if you are trading speads only and single legs are taking up most of the screen, this
            # helps save your screen space a bit).
            # We could extend this "show/hide" system to different categories or symbols in the future.
            for qp, (sym, quote) in enumerate(qs):
                if niceticker := formatTicker(quote):
                    rows.append(f"{qp:>2}) " + niceticker)

            # basically, if we've never reconnected, then only show one update count
            if self.updates == self.updatesReconnect:
                updatesFmt = f"[{self.updates:,}]"
            else:
                # else, the total CLI refresh count has diverged from the same-session reconnect count, so show both.
                # (why is this useful? our internal _data_ resets on a reconnect, so all our client-side moving averages, etc, go back
                #  to baseline with no history after a reconnect (because we don't know how long we were disconnected for technically),
                #  so having a "double count" can help show users to wait a little longer for the client-side derived metrics to catch up again).
                updatesFmt = f"[{self.updates:,}; {self.updatesReconnect:,}]"

            return HTML(
                # all these spaces look weird, but they (kinda) match the underlying column-based formatting offsets
                f"""[{self.clientId}] {str(self.now):<44}{onc} {updatesFmt}          {spxbreakers}          {openorders}    {openpositions}    {todayexecutions}      {todayclose}   ({daysInMonth} :: {daysInYear})\n"""
                + "\n".join(rows)
                + "\n"
                + balrows
            )
        except:
            logger.exception("qua?")
            return HTML("No data yet...")  # f"""{self.now:<40}\n""")

    async def qask(self, terms) -> dict[str, Any] | None:
        """Ask a questionary survey using integrated existing toolbar showing"""
        result = dict()
        extraArgs = dict(
            bottom_toolbar=self.bottomToolbar,
            refresh_interval=self.toolbarUpdateInterval,
            style=self.toolbarStyle,
        )
        for t in terms:
            if not t:
                continue

            try:
                got = await t.ask(**extraArgs)
            except EOFError:
                # if user hits CTRL-D in an input box, we get an exception which is just an input error
                got = None

            # if user canceled, give up
            # See: https://questionary.readthedocs.io/en/stable/pages/advanced.html#keyboard-interrupts
            if got is None:
                return None

            result[t.name] = got

        return result

    def levelName(self):
        if self.isSandbox is None:
            return "undecided"

        if self.isSandbox:
            return "paper"

        return "live"

    def addQuoteFromContract(self, contract):
        """Add live quote by providing a resolved contract"""
        # logger.info("Adding quotes for: {} :: {}", ordReq, contract)

        # just verify this contract is already qualified (will be a cache hit most likely)
        if isinstance(contract, Bag):
            assert all(
                [x.conId for x in contract.comboLegs]
            ), f"Sorry, your bag doesn't have qualified contracts inside of it? Got: {contract}"
        else:
            assert (
                contract.conId or contract.lastTradeDateOrContractMonth
            ), f"Sorry, we only accept qualified contracts for adding quotes, but we got: {contract}"

        # remove spaces from OCC-like symbols for consistent key reference
        symkey = lookupKey(contract)

        DELAYED = False
        # don't double-subscribe to symbols! If something is already in our quote state, we have an active subscription!
        if symkey not in self.quoteState:
            tickFields = tickFieldsForContract(contract)

            if isinstance(contract, Future):
                # enable delayed data quotes for VIX/VX/VXM quotes because they are in a non-default quote package
                # (Note: even though we mark delayed data here, I still get no results. TBD.)
                if contract.tradingClass.startswith("VX"):
                    # https://interactivebrokers.github.io/tws-api/market_data_type.html
                    DELAYED = True
                    self.ib.reqMarketDataType(3)

            # defend against some simple contracts not being qualified before reaching here
            if not contract.exchange:
                contract.exchange = "SMART"

            # logger.info("[{}] Adding new live quote: {}", symkey, contract)
            ticker = self.ib.reqMktData(contract, tickFields)
            self.quoteState[symkey] = ITicker(ticker, self)

            # Note: IBKR uses the same 'contract id' for all bags, so this is invalid for bags...
            self.contractIdsToQuoteKeysMappings[contract.conId] = symkey

            # if we enabled a delayed quote for a single reqMktData() call, return back to Live+Frozen quotes for regular requests
            if DELAYED:
                DELAYED = False
                self.ib.reqMarketDataType(2)

            # This is a nice debug helper just showing the quote key name to the attached contract subscription:
            # logger.info("[{}]: {}", symkey, contract)

            # re-comply all tickers when anything is added
            self.complyITickersSharedState()

        return symkey

    @property
    def quoteStateSorted(self):
        """Return the EXACT toolbar ticker/quote content in position-accurate iteration order.

        This can be used for iterating quotes/tickers by position if we need to elsewhere."""
        tickersSortedByPosition = sorted(
            self.quoteState.items(), key=lambda x: sortQuotes(x, self.conIdCache)
        )

        # replace the global mapping each time too
        self.quotesPositional = tickersSortedByPosition

        return tickersSortedByPosition

    def quoteExists(self, contract):
        return lookupKey(contract) in self.quoteState

    def scanStringReplacePositionsWithSymbols(self, query: str) -> str:
        """Take an input string having any number of :N references and replace them with symbol names in the output.

        e.g. "Checking :32 for updates" -> "Checking AAPL for updates"
        """

        # We match to INCLUDE the ":" because `.quoteResolve()` _strips_ the leading `:` itself.
        return re.sub(
            r"(:\d+)",
            lambda match: self.quoteResolve(match.group(1))[0] or "NOT_FOUND",
            query,
        )

    async def positionalQuoteRepopulate(
        self, sym: str, exchange: str | None = "SMART"
    ) -> tuple[str | None, Contract | None]:
        """Given a symbol request which may contain :N replacement indicators, return resolved symbol instead."""

        assert sym

        # single symbol positional request
        if sym[0] == ":":
            foundSymbol, contract = self.quoteResolve(sym)
            # assert foundSymbol and contract and contract.symbol
            return foundSymbol, contract

        # single symbol no spaces
        if " " not in sym:
            try:
                contract = contractForName(sym, exchange=exchange)
            except Exception as e:
                # Note: don't make this logger.exception() except for temporary debugging.
                #       (because logger.exception pauses for 30+ when drawing stack trace during live sessions)
                logger.error("Contract creation failed: {}", str(e))
                return None, None

            (contract,) = await self.qualify(contract)
            assert contract and contract.conId

            return sym, contract

        def symFromContract(c):
            if isinstance(c, FuturesOption):
                # Need to construct OCC-like format so the symbol
                # parser can deconstruct it back into a contract:
                # symbol[date][right][strike]
                fsym = c.symbol

                # remove the leading "20"
                fdate = c.lastTradeDateOrContractMonth[2:]
                fright = c.right
                fstrike = c.strike
                tradingClass = c.tradingClass

                tradingClassExtension = f"-{tradingClass}" if tradingClass else ""
                return f"/{fsym}{fdate}{fright}{int(fstrike * 1000):08}{tradingClassExtension}"

            return c.localSymbol.replace(" ", "")

        # a symbol request with spaces which could require replacing resolved quotes inside of it
        rebuild: list[str] = []
        for part in sym.split():
            if part[0] == ":":
                foundSymbol, contract = self.quoteResolve(part)

                # logger.info("resolved: {} {}", foundSymbol, contract)

                # if we are adding a spread, combine each leg in their current order
                # (i.e. we aren't respecting the user buy/sell request, but rather just replacing
                #       the current spread as it exists from a different quote into this quote)
                if isinstance(contract, Bag):
                    # Look up all legs of this spread/bag
                    legs = contract.comboLegs
                    contracts = await self.qualify(
                        *[Contract(conId=leg.conId) for leg in legs]
                    )

                    # remove the previous two elements of the current rebuild list because they are
                    # just the "buy 1" before this ":nn" field.
                    rebuild = rebuild[:-2]

                    # now append all leg add commands based on their sides and ratios in the spread description
                    for leg, contract in zip(legs, contracts):
                        # We need to use our futures option syntax because we pass strings to the bag constructor,
                        # but futures options symbols aren't OCC symbols
                        rebuild.append(leg.action.lower())
                        rebuild.append(str(leg.ratio))
                        rebuild.append(symFromContract(contract))
                else:
                    assert foundSymbol
                    rebuild.append(symFromContract(contract))
            else:
                rebuild.append(part)

        # now put it back together again...
        sym = " ".join(rebuild)
        logger.info("Using add request: {}", sym)
        orderReq = self.ol.parse(sym)

        return sym, await self.bagForSpread(orderReq)

    async def addQuotes(self, symbols: Iterable[str]):
        """Add quotes by a common symbol name"""
        if not symbols:
            return

        ors: list[buylang.OrderRequest] = []
        sym: str
        for sym in symbols:
            sym = sym.upper()
            # don't attempt to double subscribe
            # TODO: this only checks the named entry, so we need to verify we aren't double subscribing /ES /ESZ3 etc
            if sym in self.quoteState:
                continue

            # if this is a spread quote, attempt to replace any :N requests with the actual symbols...
            sym, _contract = await self.positionalQuoteRepopulate(sym)  # type: ignore

            # if creation failed, we can't process it...
            if not sym:
                continue

            orderReq = self.ol.parse(sym)
            ors.append(orderReq)

            # if this is a multi-part spread order, also add quotes for each leg individually!
            if orderReq.isSpread():
                for oo in orderReq.orders:
                    osym = oo.symbol
                    ors.append(self.ol.parse(osym))

        # technically not necessary for quotes, but we want the contract
        # to have the full '.localSymbol' designation for printing later.
        cs: list[Contract | None] = await asyncio.gather(
            *[self.contractForOrderRequest(o) for o in ors]
        )

        # logger.info("Resolved contracts: {}", cs)

        # the 'contractForOrderRequest' qualifies contracts before it returns, so
        # all generated contracts already have their fields populated correctly here.

        for ordReq, contract in zip(ors, cs):
            if not contract:
                logger.error(
                    "Failed to find live contract for: {} :: {}", ordReq, contract
                )
                continue

            symkey = self.addQuoteFromContract(contract)

        # check if all contracts exist in the instrumentdb (and schedule creating them if not)
        self.idb.load(*cs)

        # return contracts added
        return list(filter(None, cs))

    def complyITickersSharedState(self) -> None:
        """Iterate all subscribed tickers looking to attach bags to their legs and legs to their bags."""
        # We need to evalute all subscribed bags so their .legs match.
        # Also we need to attach each contract leg to bag(s) it belongs to.
        # This should be run after any bag addition or removal because the tickers themselves don't have
        # access to the full quote state to read other tickers (so we must manually attach related tickers).

        idToTicker = lambda x: self.quoteState.get(
            self.contractIdsToQuoteKeysMappings.get(x)  # type: ignore
        )

        # first, reset all membership, then re-add all membership...
        for symkey, iticker in self.quoteState.items():
            iticker.legs = tuple()
            iticker.bags.clear()

        # now attach each leg to its owning bags, and populate each bag with its leg tickers
        for symkey, iticker in self.quoteState.items():
            t = iticker.ticker

            if isinstance(t.contract, Bag):
                width = 0.0
                legs = list()
                for leg in t.contract.comboLegs:
                    legTicker = idToTicker(leg.conId)

                    # add this bag as being used by the current leg
                    if legTicker:
                        legTicker.bags.add(iticker)

                    # generate leg and ticker descriptors for the bag
                    match leg.action:
                        case "BUY":
                            legs.append((leg.ratio, legTicker))
                            if legTicker:
                                width += (
                                    leg.ratio
                                    * (legTicker.contract.strike or nan)
                                    * (-1.0 if legTicker.contract.right == "C" else 1.0)
                                )
                        case "SELL":
                            legs.append((-leg.ratio, legTicker))
                            if legTicker:
                                width += (
                                    leg.ratio
                                    * (legTicker.contract.strike or nan)
                                    * (-1.0 if legTicker.contract.right == "P" else 1.0)
                                )
                        case _:
                            logger.warning("Unexpected action? Got: {}", leg.action)

                # attach leg descriptors to bag ticker
                assert (
                    len(legs) == len(t.contract.comboLegs)
                ), "Why didn't we populate all legs into the legs descriptors? Our math is invalid if this happens."

                iticker.legs = tuple(legs)
                iticker.width = width
                iticker.updateGreeks()

    async def runCollective(self, concurrentCmds):
        """Given a list of commands and arguments, run them all concurrently."""

        # Run all our concurrent tasks NOW
        cmds = "; ".join([x[2] for x in concurrentCmds])
        with Timer(cmds):
            try:
                await asyncio.gather(
                    *[
                        self.dispatch.runop(
                            collectiveCmd,
                            collectiveRest[0] if collectiveRest else None,
                            self.opstate,
                        )
                        for collectiveCmd, collectiveRest, _originalFullCommand in concurrentCmds
                    ]
                )
            except:
                logger.exception("[{}] Collective command running failed?", cmds)

    async def runSingleCommand(self, cmd, rest):
        with Timer(cmd):
            try:
                await self.dispatch.runop(cmd, rest[0] if rest else None, self.opstate)
            except Exception as e:
                if self.localvars.get("bigerror"):
                    err = logger.exception
                else:
                    logger.warning(
                        "Using small exception printer. 'set bigerror yes' to enable full stack trace messages."
                    )
                    err = logger.error

                # NOTE: during FULL MARKET HOURS, printing these execptions now cause a 25 second terminal pause
                #       because loguru obtains a global Lock() against all logging when the full exception prints,
                #       and when we are doing 20,000 events per second, lots of things get delayed.
                # So, basically, set 'bigerror' if you care about full exceptions, else trust the smaller exceptions.
                se = str(e)
                if "token" in se or "terminal" in se:
                    # don't show a 100 line stack trace for mistyped inputs.
                    # Just tell the user it needs to be corrected.
                    err("[{}] Error parsing your input: {}", [cmd] + rest or [], se)
                else:
                    err("[{}] Error with command: {}", [cmd] + rest or [], se)

    def buildRunnablesFromCommandRequest(self, text1):
        # Attempt to run the command(s) submitted into the prompt.
        #
        # Commands can be:
        # Regular single-line commands:
        #  > COMMAND
        #
        # Multiple commands on a single line with semicolons splitting them:
        #  > COMMAND1; COMMAND2
        #
        # Multiple commands across multiple lines (easy for pasting from other scripts generating commands)
        #  > COMMAND1
        #    COMMAND2
        #
        # Commands can have end of line comments which *do* get saved to history, but *DO NOT* get sent to the command
        # > COMMAND # Comment about command
        #
        # Commands can also be run in groups all at once concurrently.
        # Concurrent commands requested back-to-back all run at the same time and non-concurrent commands between concurrent groups block as expected.
        #
        # This will run (1, 2) concurrently, then 3, then 4, then (5, 6) concurrently again.
        # > COMMAND1&; COMMAND2&; COMMAND3; COMMAND4; COMMAND5&; COMMAND6&
        #
        # Command processing process is:
        #  - Detect end-of-line comment and remove it (comments are PER FULL INPUT so "CMD1; CMD2; # CMD3; CMD4; CMD5" only runs "CMD1; CMD2")
        #  - Split input text on newlines and semicolons
        #  - Remove leading/trailing whitespace from each split command
        #  - Check if command is a concurrent command request (add to concurrent group if necessary)
        #  - Check if command is regular (add to regular runner if necessary)
        #  - Run collected concurrent and sequential command(s) in submitted group order.
        #
        # Originally we didn't have concurrency groups, so we processed commands in a simple O(N) loop,
        # but now we pre-process (concurrent, sequential) commands first, then we run commands after we
        # accumulate them, so we have ~O(2N) processing, but our N is almost always less than 10.
        #
        # (This command processing logic went from "just parse 1 command per run" to our
        #  current implementation of handling multi-commands and comments and concurrent commands,
        #  so command parsing has increased in complexity, but hopefully the increased running logic is
        #  useful to enable more efficient order entry/exit management.)
        #
        # These nice helpers require some extra input processing work, but our
        # basic benchmark shows cleaning up these commands only requires an
        # extra 30 us at the worst case, so it still allows over 30,000 command
        # parsing events per second (and we always end up blocked by the IBKR
        # gateway latency anyway which takes 100 ms to 300 ms for replies to the API)

        runnables: list[Awaitable[None]] = []

        # 'collective' holds the current accumulating concurrency group
        collective = []

        # Note: comments (if any) must have a leading space so we don't wipe out things like setting color hex codes with fg:#dfdfdf etc

        # We needed this more complex command runner because we can't just "split on semicolons" since if the semicolon designators
        # are *inside* quotes, we must not split commands *inside* a quote group because anything inside quotes must remain
        # untouched because it should be passed as-is as parameters for further processing.
        ccmds = split_commands(text1)
        # logger.info("ccmds: {}", ccmds)

        for ccmd in ccmds:
            ccmd = ccmd.strip()

            # if the split generated empty entries (like running ;;;;), just skip the command
            if not ccmd:
                continue

            # custom usability hack: we can detect math ops and not need to prefix 'math' to them manually
            if ccmd[0] == "(":
                ccmd = f"math {ccmd}"
            elif ccmd.startswith("if ") or ccmd.startswith("while "):
                # also allow 'if' statements directly then auto-prepend 'ifthen' to them.
                ccmd = f"ifthen {ccmd}"

            # Check if this command is a background command then clean it up
            isBackgroundCmd = ccmd[-1] == "&"
            if isBackgroundCmd:
                # remove ampersand from background request and re-strip command...
                ccmd = ccmd[:-1].rstrip()

            # split into command dispatch lookup and arguments to command
            cmd, *rest = ccmd.split(" ", 1)
            # logger.info("cmd: {}, rest: {}", cmd, rest)

            # If background command, add to our background concurrency group for this block
            if isBackgroundCmd:
                # now fixup background command...
                collective.append((cmd, rest, ccmd))

                # this 'run group' count is BEFORE the runnable is added
                logger.info(
                    "[{} :: concurrent] Added command to run group {}",
                    ccmd,
                    len(runnables),
                )
                continue

            # if we have previously saved concurrent tasks and this task is NOT concurrent, add all concurrent tasks,
            # THEN add this task.
            if collective and not isBackgroundCmd:
                runnables.append(self.runCollective(collective.copy()))

                # now since we added everything, remove the pending tasks so we don't schedule them again.
                collective.clear()

            # now schedule SINGLE command since we know the collective is properly handled already
            runnables.append(self.runSingleCommand(cmd, rest))

            if len(runnables) and len(ccmds) > 1:
                # this 'run group' count is AFTER the runnable is added (so we subtract one to get the actual order number)
                logger.info(
                    "[{} :: sequential] Added command to run group {}",
                    ccmd,
                    len(runnables) - 1,
                )

        # extra catch: if our commands END with a collective command, we need to now add them here too
        # (because the prior condition only checks if we went collective->single; but if we are ALL collective,
        #  we never trigger the "is single, cut previously collective into a full group" condition)
        if collective:
            runnables.append(self.runCollective(collective.copy()))

        return runnables

    async def runall(self):
        logger.info(
            "Using ib_async version: {} :: {}",
            ib_async.version.__version__,
            ib_async.version.__version_info__,
        )
        await self.prepare()
        await self.speak.say(say=f"Starting Client {self.clientId}!")

        while not self.exiting:
            try:
                await self.dorepl()
            except:
                logger.exception("Uncaught exception in repl? Restarting...")
                continue

    async def prepare(self):
        # Setup...

        # restore colors (if exists)
        await self.dispatch.runop("colorsload", "", self.opstate)

        # flip to enable/disable verbose ib_insync library logging
        if False:
            import logging

            ib_async.util.logToConsole(logging.INFO)

        # (default is 60 seconds which is too long if connections drop out a lot)
        # NOTE: this doesn't actually do anything except fire a 'timeoutEvent' event
        #       if there is no gateway network traffic for N seconds. It also only fires once,
        #       so we shoould reset setTimeout in the event handler if we are checking for such things.
        self.ib.setTimeout(5)

        # Attach IB events *outside* of the reconnect loop because we don't want to
        # add duplicate event handlers on every reconnect!
        # Note: these are equivalent to the pattern:
        #           lambda row: self.updateSummary(row)
        self.ib.accountSummaryEvent += self.updateSummary
        self.ib.pnlEvent += self.updatePNL
        self.ib.orderStatusEvent += self.updateOrder
        self.ib.errorEvent += self.errorHandler
        self.ib.cancelOrderEvent += self.cancelHandler
        self.ib.commissionReportEvent += self.commissionHandler
        self.ib.newsBulletinEvent += self.newsBHandler
        self.ib.tickNewsEvent += self.newsTHandler

        # We don't use these event types because ib_insync keeps
        # the objects "live updated" in the background, so everytime
        # we read them on a refresh, the values are still valid.
        # self.ib.pnlSingleEvent += self.updatePNLSingle

        # we calculate some live statistics here, and this gets called potentially
        # 5 Hz to 10 Hz because quotes are updated every 250 ms.
        # This event handler also includes a utility for writing the quotes to disk
        # for later backtest handling.
        self.ib.pendingTickersEvent += self.tickersUpdate

        # openOrderEvent is noisy and randomly just re-submits
        # already static order details as new events.
        # self.ib.openOrderEvent += self.orderOpenHandler
        self.ib.execDetailsEvent += self.orderExecuteHandler
        self.ib.positionEvent += self.positionEventHandler

        async def requestMarketData():
            logger.info("Requesting market data...")

            # We used to think this needed to be called before each new market data request, but
            # apparently it works fine now only set once up front?
            # Tell IBKR API to return "last known good quote" if outside
            # of regular market hours instead of giving us bad data.
            self.ib.reqMarketDataType(2)

            # resubscribe to active quotes
            # remove all quotes and re-subscribe to the current quote state
            logger.info("[quotes] Restoring quote state...")
            self.quoteState.clear()

            # Note: always restore snapshot state FIRST so the commands further down don't overwrite
            #       our state with only startup entries.
            with Timer("[quotes :: snapshot] Restored quote state"):
                # restore CLIENT ONLY symbols
                # run the snapshot restore by itself because it hits IBKR rate limits if run with the other restores
                loadedClientDefaultQuotes = await self.dispatch.runop(
                    "qloadsnapshot", "", self.opstate
                )

            # Only load shared quotes if we don't have a local snapshot to restore.
            # (otherwise, we end up loading the global state over our per-client state, so if a client
            #  removes a default symbol, it would _always_ get added back on restart unless we exclude these...
            #  which also make us wonder if we even need the "global" quote namespace anymore ("global on load"
            #  was from before we had per-client saved quote states)).
            # TODO: make this a callable command to "Restore defaults" if we ended up with a busted quote state.
            if not loadedClientDefaultQuotes:
                contracts: list[Stock | Future | Index] = [
                    Stock(sym, "SMART", "USD") for sym in stocks
                ]
                contracts += futures
                contracts += idxs

                with Timer("[quotes :: global] Restored quote state"):
                    # run restore and local contracts qualification concurrently
                    # logger.info("pre=qualified: {}", contracts)
                    (
                        loadedClientDefaultQuotes,
                        contractsQualified,
                    ) = await asyncio.gather(
                        # restore SHARED global symbols
                        self.dispatch.runop("qrestore", "global", self.opstate),
                        # prepare to restore COMMON symbols
                        self.qualify(*contracts),
                    )
                    # logger.info("post=qualified: {}", contractsQualified)

                with Timer("[quotes :: common] Restored quote state"):
                    for contract in contractsQualified:
                        try:
                            # logger.info("Adding quote for: {} via {}", contract, contracts)
                            self.addQuoteFromContract(contract)
                        except Exception as e:
                            logger.error(
                                "Failed to add on startup: {} ({})", contract, e
                            )

            # also, re-attach predicate data readers since any previous live data sources
            # the predicates were attached to no longer exist after the reconnect().
            await asyncio.gather(
                *[
                    self.predicateSetup(prepredicate)
                    for prepredicate in self.ifthenRuntime.predicates.values()
                ]
            )

        async def reconnect():
            # don't reconnect if an exit is requested
            if self.exiting:
                return

            # TODO: we should really find a better way of running this on startup because currently, if the
            #       IBKR gateway/API is down or unreachable, icli will never actually start since we just
            #       get stuck in this "while not connected, attempt to connect" pre-launch condition forever.
            logger.info("Connecting to IBKR API...")
            while True:
                self.connected = False

                logger.info(
                    "Total Updates: {}; Updates since last connect: {}",
                    self.updates,
                    self.updatesReconnect,
                )

                self.updatesReconnect = 0

                try:
                    # NOTE: Client ID *MUST* be 0 to allow modification of
                    #       existing orders (which get "re-bound" with a new
                    #       order id when client 0 connects—but it *only* works
                    #       for client 0)
                    # If you are using the IBKR API, it's best to *never* create
                    # orders outside of the API (TWS, web interface, mobile) because
                    # the API treats non-API-created orders differently.

                    # reset cached states on reconnect so we don't show stale data
                    self.summary.clear()
                    self.pnlSingle.clear()

                    await self.ib.connectAsync(
                        self.host,
                        self.port,
                        clientId=self.clientId,
                        readonly=False,
                        account=self.accountId,
                        fetchFields=ib_async.StartupFetchALL
                        & ~ib_async.StartupFetch.EXECUTIONS,
                    )

                    logger.info(
                        "Connected! Current Request ID for Client {}: {} :: Current Server Version: {}",
                        self.clientId,
                        self.ib.client._reqIdSeq,
                        self.ib.client.serverVersion(),
                    )

                    self.connected = True

                    self.ib.reqNewsBulletins(True)

                    # we load executions fully async after the connection happens because
                    # the fetching during connection causes an extra delay we don't need.
                    self.task_create("load executions", self.loadExecutions())

                    # also load market data async for quicker non-blocking startup
                    self.task_create("req mkt data", requestMarketData())

                    # request live updates (well, once per second) of account and position values
                    self.ib.reqPnL(self.accountId)

                    # Subscribe to realtime PnL updates for all positions in account
                    # Note: these are updated once per second per position! nice.
                    # TODO: add this to the account order/filling notifications too.
                    for p in self.ib.portfolio():
                        self.pnlSingle[p.contract.conId] = self.ib.reqPnLSingle(
                            self.accountId, "", p.contract.conId
                        )

                    # run some startup accounting subscriptions concurrently
                    await asyncio.gather(
                        self.ib.reqAccountSummaryAsync(),  # self.ib.reqPnLAsync()
                    )

                    break
                except (
                    ConnectionRefusedError,
                    ConnectionResetError,
                    OSError,
                    asyncio.TimeoutError,
                    asyncio.CancelledError,
                ) as e:
                    # Don't print full network exceptions for just connection errors
                    logger.error(
                        "[{}] Failed to connect to IB Gateway, trying again... (also check this client id ({}) isn't already connected)",
                        str(e),
                        self.clientId,
                    )
                except:
                    # Do print exception for any unhandled or unexpected errors while connecting.
                    logger.exception("why?")

                try:
                    await asyncio.sleep(3)
                except:
                    logger.warning("Exit requested during sleep. Goodbye.")
                    sys.exit(0)

        try:
            # Run the initial connect in the background so it still starts up at least even if there's
            # no active server running.
            # Note: we need to run the initial connect BLOCKING because the initial connect tell us things like "is this live or sandbox,"
            #       which we use for configuring some other systems (like the history cache and process name and prompt prefix), though
            #       we could just use the same history file for all runs anyway and the prompt would fix itself.
            # asyncio.xreate_task(reconnect())
            await reconnect()
        except SystemExit:
            # do not pass go, do not continue, throw the exit upward
            sys.exit(0)

        customName = ""
        if self.customName:
            customName = f" — {self.customName}"

        set_title(f"{self.levelName().title()} Trader ({self.clientId}){customName}")
        self.ib.disconnectedEvent += lambda: self.task_create("reconnect", reconnect())

    async def buildAndRun(self, text1):
        # 'runnables' is the list of all commands to run after we collect them
        runnables = self.buildRunnablesFromCommandRequest(text1)

        # if no commands, just draw the prompt again
        if not runnables:
            return

        if len(runnables) == 1:
            # if only one command, don't run with an extra Timer() report like we do below
            # with multiple commands (individual commands always report their individual timing)
            return await runnables[0]
        else:
            # only show the "All commands" timer if we have multiple commands to run
            with Timer("All commands"):
                for run in runnables:
                    try:
                        # run a COLLECTIVE COMMAND GROUP we previously created
                        await run
                    except:
                        logger.exception("[{}] Runnable failed?", run)

    async def dorepl(self):
        session: PromptSession = PromptSession(
            history=ThreadedHistory(
                FileHistory(
                    os.path.expanduser(f"~/.tplatcli_ibkr_history.{self.levelName()}")
                )
            ),
            auto_suggest=AutoSuggestFromHistory(),
        )

        app = session.app
        loop = asyncio.get_event_loop()

        async def updateToolbar():
            """Update account balances"""
            try:
                app.invalidate()
            except:
                # network error, don't update anything
                pass

            loop.call_later(
                self.toolbarUpdateInterval, lambda: asyncio.create_task(updateToolbar())
            )

        loop.create_task(updateToolbar())

        # The Command Processing REPL
        while True:
            try:
                # read input from Prompt Toolkit
                text1 = await session.prompt_async(
                    f"{self.levelName()}> ",
                    enable_history_search=True,
                    bottom_toolbar=self.bottomToolbar,
                    # NOTE: refresh interval is handled manually by "call_later(timeout, fn)" at the end of each toolbar update
                    # refresh_interval=3,
                    # mouse_support=True,
                    # completer=completer, # <-- causes not to be full screen due to additional dropdown space
                    complete_in_thread=True,
                    complete_while_typing=True,
                    search_ignore_case=True,
                    style=self.toolbarStyle,
                )

                # log user input to our active logfile(s)
                logger.trace("{}> {}", self.levelName(), text1)

                await self.buildAndRun(text1)
            except KeyboardInterrupt:
                # Control-C pressed. Try again.
                continue
            except EOFError:
                # Control-D pressed
                logger.error("Exiting...")
                self.exiting = True
                break
            except BlockingIOError:
                # this is noisy macOS problem if using a non-fixed
                # uvloop and we don't care, but it will truncate or
                # duplicate your output.
                # solution: don't use uvloop or use a working uvloop
                try:
                    logger.error("FINAL\n")
                except:
                    pass
            except Exception:
                while True:
                    try:
                        logger.exception("Trying...")
                        break
                    except Exception:
                        await asyncio.sleep(1)
                        pass

    def task_create(self, name, coroutine, *args, **kwargs):
        # provide a default US/Eastern timezone to the scheduler unless user provides their own scheduler
        if "scheduler" not in kwargs:
            kwargs = dict(schedule=BGSchedule(tz=USEastern)) | kwargs

        return self.tasks.create(name, coroutine, *args, **kwargs)

    def task_stop(self, task):
        return self.tasks.stop(task)

    def task_stop_id(self, taskId):
        return self.tasks.stopId(taskId)

    def task_report(self):
        return self.tasks.report()

    def stop(self):
        self.exiting = True
        self.ib.disconnect()

    async def setup(self):
        pass
