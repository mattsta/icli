#!/usr/bin/env python3

original_print = print
import asyncio
import bisect
import datetime
import decimal
import fnmatch  # for glob string matching!
import locale  # automatic money formatting
import logging
import math
import os
import pathlib
import re
import statistics
import sys

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import bs4
import warnings

# http://www.grantjenks.com/docs/diskcache/
import diskcache

import numpy as np
import orjson
import pandas as pd
import pendulum

from prompt_toolkit import Application, print_formatted_text, PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, ThreadedHistory
from prompt_toolkit.shortcuts import set_title

import icli.awwdio as awwdio

import icli.calc
import icli.orders as orders

from icli.futsexchanges import FUTS_EXCHANGE

# from . import agent
# from . import accum
from .tinyalgo import ATRLive

locale.setlocale(locale.LC_ALL, "")

import ib_async

import seaborn
from ib_async import (
    Bag,
    ComboLeg,
    Contract,
    Future,
    IB,
    Index,
    NewsBulletin,
    NewsTick,
    Order,
    PnLSingle,
    RealTimeBarList,
    Ticker,
    Trade,
)

from loguru import logger

import icli.lang as lang
from icli.helpers import *  # FUT_EXP and isset() is appearing from here
import prettyprinter as pp
import tradeapis.buylang as buylang
import tradeapis.rounder as rounder
import tradeapis.cal as mcal

from cachetools import cached, TTLCache

from mutil.numeric import fmtPrice, fmtPricePad
from mutil.timer import Timer

warnings.filterwarnings("ignore", category=bs4.MarkupResemblesLocatorWarning)
pp.install_extras(["dataclasses"], warn_on_error=False)

# global client ID for your IBKR gateway connection (must be unique per client per gateway)
ICLI_CLIENT_ID = int(os.getenv("ICLI_CLIENT_ID", 0))

# environment 1 true; 0 false; flag for determining if EVERY QUOTE (4 Hz per symbol) is saved to a file
# for later backtest usage or debugging (note: this uses the default python 'json' module which sometimes
# outputs non-JSON compliant NaN values, so you may need to filter those out if read back using a different
# json parser)
ICLI_DUMP_QUOTES = bool(int(os.getenv("ICLI_DUMP_QUOTES", 0)))
ICLI_AWWDIO_URL = awwdio.ICLI_AWWDIO_URL

# Configure logger where the ib_insync live service logs get written.
# Note: if you have weird problems you don't think are being exposed
# in the CLI, check this log file for what ib_insync is actually doing.
LOGDIR = pathlib.Path(os.getenv("ICLI_LOGDIR", "runlogs"))
LOGDIR.mkdir(exist_ok=True)
LOG_FILE_TEMPLATE = str(
    LOGDIR / f"icli-id={ICLI_CLIENT_ID}-{pendulum.now('US/Eastern')}".replace(" ", "_")
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


def invertstr(x):
    return x.translate(ATOZTOA_TABLE)


# allow these values to be cached for 10 hours
@cached(cache=TTLCache(maxsize=128, ttl=60 * 60 * 10))
def fetchDateTimeOfEndOfMarketDay():
    """Return the market (start, end) timestamps for the next two market end times."""
    found = mcal.getMarketCalendar(
        "NASDAQ",
        start=pd.Timestamp("now"),
        stop=pd.Timestamp("now") + pd.Timedelta(7, "D"),
    )

    # format returned is two columns of [MARKET OPEN, MARKET CLOSE] timestamps per date.
    soonestStart = found.iat[0, 0]
    soonestEnd = found.iat[0, 1]

    nextStart = found.iat[1, 0]
    nextEnd = found.iat[1, 1]

    return [(soonestStart, soonestEnd), (nextStart, nextEnd)]


# expire this cache once every 15 minutes so we only have up to 15 minutes of wrong dates after EOD
@cached(cache=TTLCache(maxsize=128, ttl=60 * 15))
def fetchEndOfMarketDay():
    """Return the timestamp of the next end-of-day market timestamp.

    This is currently only used for showing the "end of day" countdown timer in the toolbar,
    so it's okay if we return an expired date for a little while (the 15 minute cache interval),
    so the toolbar will just report a negative closing time for up to 15 minutes.

    The cache structure is because the toolbar refresh code is called anywhere from 1 to 10 times
    _per second_ so we want to minimize as much math and logic overhead as possible for non-changing
    values.

    We could potentially place an event timer somewhere to manually clear the cache at EOD,
    but we just aren't doing it yet."""
    [(soonestStart, soonestEnd), (nextStart, nextEnd)] = fetchDateTimeOfEndOfMarketDay()

    # this logic just helps us across the "next day" barrier when this runs right after a normal 4pm close
    # so we immediately start ticking down until the next market day close (which could be 3-4 days away depending on holidays!)
    if soonestEnd > pendulum.now():
        return pendulum.from_timestamp(soonestEnd.timestamp())

    return pendulum.from_timestamp(nextEnd.timestamp())


# Fields updated live for toolbar printing.
# Printed in the order of this list (the order the dict is created)
# Some math and definitions for values:
# https://www.interactivebrokers.com/en/software/tws/usersguidebook/realtimeactivitymonitoring/available_for_trading.htm
# https://ibkr.info/node/1445
LIVE_ACCOUNT_STATUS = [
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

# we need to add this back for the CHECKS, but we don't show the BuyingPower key directly...
STATUS_FIELDS_PROCESS = set(LIVE_ACCOUNT_STATUS) | {"BuyingPower"}


def readableHTML(html):
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

# logger.info("futures are: {}", futures)


@dataclass
class IBKRCmdlineApp:
    # Your IBKR Account ID (required)
    accountId: str

    # number of seconds between refreshing the toolbar quote/balance views
    # (more frequent updates is higher redraw CPU utilization)
    toolbarUpdateInterval: float = 2.22

    host: str = "127.0.0.1"
    port: int = 4001

    clientId: int = ICLI_CLIENT_ID

    # initialized to True/False when we first see the account
    # ID returned from the API which will tell us if this is a
    # sandbox ID or True Account ID
    isSandbox: bool | None = None

    # The Connection
    ib: IB = field(default_factory=IB)

    # count total toolbar refreshes
    updates: int = 0

    # True if use sound for trades...
    alert: bool = False

    # Events!
    scheduler: dict[str, Any] = field(default_factory=dict)

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
    quoteState: dict[str, Ticker] = field(default_factory=dict)
    contractIdsToQuoteKeysMappings: dict[int, str] = field(default_factory=dict)
    depthState: dict[Contract, Ticker] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)
    position: dict[str, float] = field(default_factory=dict)
    order: dict[str, float] = field(default_factory=dict)
    liveBars: dict[str, RealTimeBarList] = field(default_factory=dict)
    pnlSingle: dict[int, PnLSingle] = field(default_factory=dict)
    exiting: bool = False
    ol: buylang.OLang = field(default_factory=buylang.OLang)
    quotehistory: dict[int, deque[float]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=120))
    )

    # calculate live ATR based on quote updates
    # (the .25 is because quotes update at 250 ms intervals, so we normalize "events per second" by update frequency)
    atrs: dict[str, ATRLive] = field(
        default_factory=lambda: defaultdict(
            lambda: ATRLive(int(90 / 0.25), int(45 / 0.25))
        )
    )

    speak: awwdio.AwwdioClient = field(default_factory=awwdio.AwwdioClient)

    # hold EMA per current symbol with various lookback periods
    ema: dict[str, dict[int, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )

    # Specific dict of ONLY fields we show in the live account status toolbar.
    # Saves us from sorting/filtering self.summary() with every full bar update.
    accountStatus: dict[str, float] = field(
        default_factory=lambda: dict(
            zip(LIVE_ACCOUNT_STATUS, [0.00] * len(LIVE_ACCOUNT_STATUS))
        )
    )

    # Cache all contractIds and names to their fully qualified contract object values
    conIdCache: diskcache.Cache = field(
        default_factory=lambda: diskcache.Cache("./cache-contracts")
    )

    connected: bool = False
    disableClientQuoteSnapshotting: bool = False
    loadingCommissions: bool = False

    def __post_init__(self) -> None:
        # just use the entire IBKRCmdlineApp as our app state!
        self.opstate = self

        # provide ourself to the calculator so the calculator can lookup live quote prices and live account values
        self.calc = icli.calc.Calculator(self)

    async def qualify(self, *contracts) -> list[Contract]:
        """Qualify contracts against the IBKR allowed symbols.

        Mainly populates .localSymbol and .conId

        We also cache the results for ease of re-use and for mapping
        contractIds back to names later."""

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

        # Our cache operates on two tiers:
        #  - ideally, we look up contracts by id directly
        #  - alternatively, we look up contracts by name, but we also need to know the _type_ of the name we are looking up.
        # So, we check the requested contracts for:
        #  - if input contract already has a contract id, we look up the conId directly.
        #  - if input contract doesn't have an id, we generate a lookup key of Class-Symbol like "Future-ES" or "Index-SPX"
        #    so we can retrieve the correct instrument class combined with the proper symbol from a cached contract.
        for contract in contracts:
            cached_contract = None
            # logger.info("Looking up: {} :: {}", contract, contractToSymbolDescriptor(contract))

            try:
                # only attempt to look up using ID if ID exists, else attempt to lookup by name
                if contract.conId:
                    # Attempt first lookup using direct ID, but if ID isn't found try to use the Class-Symbol key format...
                    cached_contract = self.conIdCache.get(contract.conId)  # type: ignore
                else:
                    cached_contract = self.conIdCache.get(
                        contractToSymbolDescriptor(contract)
                    )  # type: ignore

                # logger.info("Using cached contract for {}: {} :: {}", contract.conId, cached_contract, contract)
            except ModuleNotFoundError:
                # the pickled contract is from another library we don't have loaded in this environment anymore,
                # so we need to drop the existing bad pickle and re-save it
                try:
                    del self.conIdCache[contract.conId]
                    del self.conIdCache[contractToSymbolDescriptor(contract)]
                except:
                    pass

            # if we _found_ a contract (and the contract has an id (just defensively preventing invalid contracts in the cache)),
            # then we don't look it up again.
            if cached_contract and cached_contract.conId:
                # logger.info("Found in cache: {} for {}", cached_contract, contract)
                cached_contracts[cached_contract.conId] = cached_contract
                totalResult[id(contract)] = cached_contract
            else:
                # else, we need to look up this contract before returning.
                # logger.info("Not found in cache: {}", contract)
                uncached_contracts.append(contract)

                # also populate unresolved contract for safety in case it can't be resolved
                # we just return it directly as originally provided
                totalResult[id(contract)] = contract

        # logger.info("CACHED: {} :: UNCACHED: {}", cached_contracts, uncached_contracts)

        # if we have NO UNCACHED CONTRACTS, then we found all input requests in the cache,
        # so we can just return the cached contracts found directly (and they are already
        # in the correct input-return order).
        if not uncached_contracts:
            return list(cached_contracts.values())

        # For uncached, fetch them from the IBKR lookup system
        got = []
        try:
            # logger.info("Looking up uncached contracts: {}", uncached_contracts)
            got = await asyncio.wait_for(
                self.ib.qualifyContractsAsync(*uncached_contracts), timeout=2
            )
        except Exception as e:
            logger.error(
                "Timeout while trying to qualify {} contracts (sometimes IBKR is slow or the API is offline during nightly restarts) :: {}",
                len(uncached_contracts),
                str(e),
            )

        # iterate resolved contracts and cache them by multiple lookup keys
        for contract in got:
            # the `qualifyContractsAsync` modifies the contracts in-place, so we map their
            # id to itself since we replaced it directly.
            # (yes, we _always_ set this even if we didn't resolve a 'conId' because we need
            #  to return _all_ contracts back to the user in the order of their inputs, so
            #  we need every input contract to be in the 'totalResult' map regardless of its final
            #  success/fail resolution value)
            totalResult[id(contract)] = contract

            # Only cache actually qualified contracts with a full IBKR contract ID
            if not contract.conId:
                continue

            cached_contracts[contract.conId] = contract

            # we want Futures contracts to refresh more often because they have
            # embedded expiration dates which may change over time if we are using
            # generic symbol names like "ES" for the next main contract.
            EXPIRATION_DAYS = 2 if isinstance(contract, Future) else 30

            # cache by id
            # logger.info("Saving {} -> {}", contract.conId, contract)
            self.conIdCache.set(
                contract.conId, contract, expire=86400 * EXPIRATION_DAYS
            )  # type: ignore

            # also set by Class-Symbol designation as key (e.g. "Index-SPX" or "Future-ES")
            # logger.info("Saving {} -> {}", contractToSymbolDescriptor(contract), contract)
            self.conIdCache.set(
                contractToSymbolDescriptor(contract),
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
            self.localvars[key] = val
        else:
            # else, if value not provided, remove key (if exists; not an error if key doesn't exist)
            self.localvars.pop(key)

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
        self, oreq: buylang.OrderRequest, exchange=None
    ) -> Contract | None:
        """Return a valid qualified contract for any order request.

        If order request has multiple legs, returns a Bag contract representing the spread.
        If order request only has one symbol, returns a regular future/stock/option contract.

        If symbol(s) in order request are not valid, returns None."""

        if not exchange:
            exchange = self.localvars.get("exchange", "SMART")

        if oreq.isSpread():
            return await self.bagForSpread(oreq, exchange)

        if oreq.isSingle():
            contract = contractForName(oreq.orders[0].symbol, exchange=exchange)
            cgot: list[Contract] = await self.qualify(contract)

            # only return success if the contract validated
            if cgot and cgot[0].conId:
                return cgot[0]

            return None

        # else, order request had no orders...
        return None

    async def bagForSpread(
        self, oreq: buylang.OrderRequest, exchange=None, currency="USD"
    ) -> Bag | None:
        """Given a multi-leg OrderRequest, return a qualified Bag contract.

        If legs do not validate, returns None and prints errors along the way."""

        if not exchange:
            exchange = self.localvars.get("exchange", "SMART")

        # For IBKR spreads ("Bag" contracts), each leg of the spread is qualified
        # then placed in the final contract instead of the normal approach of qualifying
        # the final contract itself (because Bag contracts have Legs and each Leg is only
        # a contractId we have to look up via qualify() individually).
        contracts = [
            contractForName(s.symbol, exchange=exchange, currency=currency)
            for s in oreq.orders
        ]
        await self.qualify(*contracts)

        if not all([c.conId for c in contracts]):
            logger.error("Not all contracts qualified!")
            return None

        # trying to match logic described at https://interactivebrokers.github.io/tws-api/spread_contracts.html
        underlyings = ",".join(sorted({x.symbol for x in contracts}))

        # Iterate (in MATCHED PAIRS) the resolved contracts with their original order details
        legs = []

        # We use more explicit exchange mapping here since future options
        # require naming their exchanges instead of using SMART everywhere.
        useExchange: str
        for c, o in zip(contracts, oreq.orders):
            useExchange = c.exchange
            leg = ComboLeg(
                conId=c.conId,
                ratio=o.multiplier,
                action="BUY" if o.isBuy() else "SELL",
                exchange=c.exchange,
            )

            legs.append(leg)

        return Bag(
            symbol=underlyings,
            exchange=useExchange or exchange,
            comboLegs=legs,
            currency=currency,
        )

    def symbolNormalizeIndexWeeklyOptions(self, name: str) -> str:
        """Weekly index options have symbol names with 'W' but orders are placed without."""
        # TODO: figure out if this is still required or not
        return name.replace("SPXW", "SPX").replace("RUTW", "RUT").replace("NDXP", "NDX")

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
            quote = self.quotesPositional[lookupInt]
            ticker = quote[1]
        except:
            # either the input wasn't ':number' or the index doesn't exist...
            return None, None

        # now we passed the integer extraction and the quote lookup, so return the found symbol for the lookup id
        assert ticker.contract
        name = (ticker.contract.localSymbol or ticker.contract.symbol).replace(" ", "")

        return name, ticker.contract

    async def placeOrderForContract(
        self,
        sym: str,
        isLong: bool,
        contract: Contract,
        qty: PriceOrQuantity,
        limit: float | bool,
        orderType: str,
        preview: bool = False,
        bracket: Bracket | None = None,
    ):
        """Place a BUY (isLong) or SELL (!isLong) for qualified 'contract' at qty/price.

        The 'qty' parameter allows switching between price amounts and share/contract/quantity amounts directly.
        """

        # Always overwrite the contract cache with our current exchange (and fallback to SMART if none are specified)
        if isinstance(contract, Future):
            # (for futures, exchange must be routed to the exact futures exchange. "SMART" routing is invalid for any futures.
            #  this also means we may crash here if the futures symbol isn't in our lookup table cache, so it would need a refresh)
            fe = FUTS_EXCHANGE[contract.symbol]
            contract.exchange = fe.exchange
            logger.info(
                "[{} :: {}] Using exchange: {}", contract.symbol, fe.name, fe.exchange
            )
        else:
            globalExchange = self.localvars.get("exchange", "SMART")
            contract.exchange = globalExchange

        # Immediately ask to add quote to live quotes for this trade positioning...
        # turn option contract lookup into non-spaced version
        sym = sym.replace(" ", "")

        if limit:
            limit = comply(contract, limit)

        if qty.is_quantity and not limit:
            logger.info("[{}] Request to order qty {} at current prices", sym, qty)
        else:
            logger.info(
                "[{}] Request to order at dynamic qty/price: {} price {:,.2f}",
                sym,
                qty,
                limit,
            )

        quotesym = sym
        # TODO: check if symbol already exists as a value from
        # while not (currentQuote := self.currentQuote(quoteKey))
        # to avoid the extra/noop add lookup here.
        if not isinstance(contract, Bag):
            await self.dispatch.runop("add", f'"{quotesym}"', self.opstate)

        if not contract.conId:
            # spead contracts don't have IDs, so only reject if NOT a spread here.
            if not isinstance(contract, Bag):
                logger.error(
                    "[{} :: {}] Not submitting order because contract not qualified!",
                    sym,
                    quotesym,
                )
                return None

        # REL and LMT/MKT/MOO/MOC orders can be outside RTH
        outsideRth = True

        # hack for Bags not having multipliers at their top level. we should probably fix this better.
        multiplier = (
            100 if isinstance(contract, Bag) else float(contract.multiplier or 1)
        )

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
            # TODO: cleanup, also verify how we want to run FAST or EVICT outside RTH?
            # Algos can only operate RTH:
            outsideRth = False

        if not (isinstance(contract, Option) or outsideRth):
            logger.warning(
                "[{}] ALGO NOT SUPPORTED FOR ALL HOURS. ORDER RESTRICTED TO RTH ONLY!",
                orderType,
            )

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

        determinedQty = None

        # if input is quantity, use quantity directly
        # TODO: also allow quantity trades to submit their own limit price like 100@3.33???
        # Maybe even "100@3.33+" to start with _our_ limit price, but then run our price-follow-tracking algo
        # if the initial offer doesn't execute after a couple seconds?
        if qty.is_quantity:
            determinedQty = qty.qty

        # Also, this loop does quote lookup to generate the 'limit' price if none exists.
        # Conditions:
        #  - if quantity is a dollar amount, we need to calculate quantity based on current quote.
        #  - also, if this is a preview (with or without a limit price), we calculate a price for margin calculations.
        #  - basically: guard against quantity orders attempting to lookup prices when they aren't needed.
        #    (market orders also imply quantity is NOT money because a market order with no quantity doesn't make sense)
        if (not limit and "MKT" not in orderType) or preview:
            quoteKey = lang.lookupKey(contract)

            # if this is a new quote just requested, we may need to wait
            # for the system to populate it...
            loopFor = 10
            while not (currentQuote := self.currentQuote(quoteKey, show=True)):
                logger.warning(
                    "[{} :: {}] Waiting for quote to populate...", quoteKey, loopFor
                )
                try:
                    await asyncio.sleep(0.033)
                except:
                    logger.warning("Cancelled waiting for quote...")
                    return

                if (loopFor := loopFor - 1) == 0:
                    # if we exhausted the loop, we didn't get a usable quote so we can't
                    # do the requested price-based position sizing.
                    logger.error("Never received valid quote prices: {}", currentQuote)
                    return

            bid, ask = currentQuote

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

            if bid == -1:
                logger.warning(
                    "[{}] WARNING: No bid price, so just using ASK directly for buying!",
                    quoteKey,
                )
                bid = ask

            # Note: this logic is different than the direct 'evict' logic where we place wider limit
            #       bounds in an attempt to get out as soon as possible. This is more "at market, best effort,
            #       and follow the price if we don't get it the first time" attempts.
            if isinstance(contract, Option):
                # Options retain "regular" midpoint behavior because spreads can be wide and hopefully
                # quotes are fairly slow/stable.
                mid = round(((bid + ask) / 2), 2)

                # if no bid (nan), just play off the ask.
                if mid != mid:
                    mid = round(ask / 2, 2)
            else:
                # equity, futures, etc get the wider margins
                # NOTE: this looks backwards because for us to ACQUIRE a psoition we must be BETTER than the market
                #       on limit prices, so here we have BUY LOW and SELL HIGH just to get the position at first.
                # TODO: these offsets need to be more adaptable to recent ATR-like conditions per symbol,
                #       but the goal here is immediate fills at market-adjusted prices anyway.
                mid = round(((bid + ask) / 2) * (1.005 if isLong else 0.995), 2)

            # only use our automatic-midpoint if we don't already have a limit price
            if not limit:
                limit = comply(contract, mid)

        # only update qty if this is a money-ask because we also use this limit discovery
        # for quantity-only orders, where we don't want to alter the quantity, obviously.
        if qty.is_money:
            amt = qty.qty

            # calculate order quantity for spend budget at current estimated price
            logger.info("[{}] Trying to order ${:,.2f} worth at {}...", sym, amt, qty)

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
                determinedQty * limit * multiplier,
            )

        # declare default values so we can check against them later...
        profitOrder = None
        lossOrder = None

        try:
            sideOpen = "BUY" if isLong else "SELL"
            sideClose = "SELL" if isLong else "BUY"

            logger.info(
                "[{} :: {}] {:,.2f} @ ${:,.2f} x {:,.2f} (${:,.2f}) ALL_HOURS={} TIF={}",
                orderType,
                sideOpen,
                determinedQty,
                limit,
                multiplier,
                determinedQty * limit * multiplier,
                outsideRth,
                tif,
            )

            order = orders.IOrder(
                sideOpen,
                determinedQty,
                limit,
                outsiderth=outsideRth,
                tif=tif,
            ).order(orderType)

            if bracket:
                # When creating attached orders, we need manual order IDs because by default they only
                # get generated during the order placement phase.
                order.orderId = self.ib.client.getReqId()
                order.transmit = False

                if bracket.profitLimit is not None:
                    profitOrder = orders.IOrder(
                        sideClose,
                        determinedQty,
                        bracket.profitLimit,
                        outsiderth=outsideRth,
                        tif=tif,
                    ).order(bracket.orderProfit)

                    profitOrder.orderId = self.ib.client.getReqId()
                    profitOrder.parentId = order.orderId
                    profitOrder.transmit = False

                if bracket.lossLimit is not None:
                    lossOrder = orders.IOrder(
                        sideClose,
                        determinedQty,
                        bracket.lossLimit,
                        aux=bracket.lossStop,
                        outsiderth=outsideRth,
                        tif=tif,
                    ).order(bracket.orderLoss)

                    lossOrder.orderId = self.ib.client.getReqId()
                    lossOrder.parentId = order.orderId
                    lossOrder.transmit = False

                # if loss order exists, it ALWAYS transmits last
                if lossOrder:
                    lossOrder.transmit = True
                elif profitOrder:
                    # else, only PROFIT ORDER exists, so send it (profit order is ignored)
                    profitOrder.transmit = True

        except:
            logger.exception("ORDER GENERATION FAILED. CANNOT PLACE ORDER!")
            return

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

        name = contract.localSymbol.replace(" ", "")
        desc = f"{name} :: QTY {order.totalQuantity:,}"
        if preview:
            # Also note: there is _something_ off with our math because we aren't getting exactly 30% or 25% or 3% or 5% etc,
            #            but it's close enough for what we're trying to show at this point.
            previewPrice = order.lmtPrice if isset(order.lmtPrice) else limit

            logger.info(
                "[{}] PREVIEW REQUEST {} via {}",
                desc,
                contract,
                pp.pformat(order),
            )
            try:
                trade = await asyncio.wait_for(
                    self.ib.whatIfOrderAsync(contract, order), timeout=2
                )
            except:
                logger.error(
                    "Timeout while trying to run order preview (sometimes IBKR is slow or the order preview API could be offline)"
                )
                return None

            logger.info("[{}] PREVIEW RESULT: {}", desc, pp.pformat(trade))

            if not trade:
                logger.error("Preview not created for order?")
                return False

            # We currently assume only two kinds of things exist. We could add more.
            nameOfThing = "SHARE" if isinstance(contract, Stock) else "CONTRACT"

            # set 100% margin defaults so our return value has something populated even if margin isn't relevant (options, etc)
            margPctInit = 100
            margPctMaint = 100
            multiplier = float(contract.multiplier or 1)

            # for options or other conditions, there's no margin change to report.
            # also, if there is a "warning" on the trade, the numbers aren't valid.
            # Also, we need this extra 'isset()' check because unpopulated values from IBKR show up as string '1.7976931348623157E308'
            if (
                not (trade.warningText)
                and float(trade.initMarginChange) > 0
                and isset(float(trade.initMarginChange))
                and previewPrice
            ):
                baseTotal = order.totalQuantity * previewPrice * multiplier
                margPctInit = (float(trade.initMarginChange) / baseTotal) * 100

                margPctMaint = (float(trade.maintMarginChange) / baseTotal) * 100

                logger.info(
                    "[{}] PREVIEW MARGIN REQUIREMENT INIT: {:.2f} % (${:,.2f})",
                    desc,
                    margPctInit,
                    float(trade.initMarginChange),
                )

                # "MAIN" for "MAINTENANCE" to match the length of "INIT" above for alignment.
                logger.info(
                    "[{}] PREVIEW MARGIN REQUIREMENT MAIN: {:.2f} % (IBKR is loaning {:.2f} %)",
                    desc,
                    margPctMaint,
                    100 - margPctMaint,
                )

                logger.info(
                    "[{}] PREVIEW INIT MARGIN PER {}: ${:,.2f}",
                    desc,
                    nameOfThing,
                    float(trade.initMarginChange) / order.totalQuantity,
                )

            # don't print floats if not necessary
            imul = int(multiplier)
            if imul == multiplier:
                multiplier = imul

                # temporary (?) hack/fix for bags not having multiplers themselves, so we assume we're doing spreads of 100 multiplier option legs currently
                if isinstance(contract, Bag):
                    multiplier = 100

            # TODO: it woudl be nice to use the symbol's actual .minTick here, but it's extra work to fetch the ContractDetails itself.
            # (why these intervals? some products trade only in 0.05 increments, others in 0.10 increments, some trade in 0.25 increments; some 0.01, etc)
            #
            # ADD TO PREVIEW: calculation for current quote spread (bounce-out loss immediately as percentage of buy, if wait for 2 ticks down and want to exit, that's 3+ ticks of opposite side, etc)
            #                 also include: a {10%, 30%, 50%} loss is $X drop in contract...

            leverageKind = (
                "CONTRACT"
                if isinstance(contract, (Bag, Option, Future, FuturesOption))
                else "STOCK"
            )

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

            # "MAIN" for "MAINTENANCE" to match the length of "INIT" above for alignment.
            if isset(trade.minCommission):
                # options and stocks have a range of commissions
                logger.info(
                    "[{}] PREVIEW COMMISSION PER {}: ${:.4f} to ${:.4f}",
                    desc,
                    nameOfThing,
                    (trade.minCommission) / order.totalQuantity,
                    (trade.maxCommission) / order.totalQuantity,
                )

                if multiplier > 1:
                    # (Basically: how much must the underlying change in price for you to pay off the commission for this order.
                    tcmin = trade.minCommission / order.totalQuantity / multiplier
                    tcmax = trade.maxCommission / order.totalQuantity / multiplier
                    logger.info(
                        "[{}] PREVIEW COMMISSION PER UNDERLYING: ${:.4f} to ${:.4f} (2x: ${:.4f} to ${:.4f})",
                        desc,
                        tcmin,
                        tcmax,
                        2 * tcmin,
                        2 * tcmax,
                    )
            elif isset(trade.commission):
                # futures contracts and index options contracts have fixed priced commissions so
                # they don't provide a min/max range, it's just one guaranteed value.
                logger.info(
                    "[{}] PREVIEW COMMISSION PER CONTRACT: ${:.4f}",
                    desc,
                    (trade.commission) / order.totalQuantity,
                )

                tc = trade.commission / order.totalQuantity / multiplier
                if multiplier > 1:
                    logger.info(
                        "[{}] PREVIEW COMMISSION PER UNDERLYING: ${:.4f} (2x: ${:.4f})",
                        desc,
                        tc,
                        2 * tc,
                    )

            # calculate percentage width of the spread just to note if we are trading difficult to close positions
            spreadDiff = ((ask - bid) / bid) * 100
            logger.info(
                "[{}] BID/ASK SPREAD IS {:,.2f} % WIDE (${:,.2f} spread @ ${:,.2f} total)",
                desc,
                spreadDiff,
                ask - bid,
                (determinedQty * limit * (spreadDiff / 100) * multiplier),
            )

            if spreadDiff > 5:
                logger.warning(
                    "[{}] WARNING: BID/ASK SPREAD ({:,.2f} %) MAY CAUSE NOTICEABLE LOSS/SLIPPAGE ON EXIT",
                    desc,
                    spreadDiff,
                )

            # TODO: make this delta range configurable? config file? env? global setting?
            if isinstance(contract, (Option, FuturesOption)):
                delta = self.quoteState[name].modelGreeks.delta
                if not delta:
                    logger.warning("[{}] WARNING: OPTION DELTA NOT POPULATED YET", desc)
                elif abs(delta) <= 0.15:
                    logger.warning(
                        "[{}] WARNING: OPTION DELTA IS LOW ({:.2f}) — THIS MAY NOT WORK FOR SHORT TERM TRADING",
                        desc,
                        delta,
                    )

            # (if trade isn't valid, trade is an empty list, so only print valid objects...)
            if trade:
                # sigh, these are strings of course.
                excess = float(trade.equityWithLoanAfter) - float(trade.initMarginAfter)
                if excess < 0:
                    logger.warning(
                        "[{}] TRADE NOT VIABLE. MISSING EQUITY: ${:,.2f}",
                        desc,
                        excess,
                    )
                else:
                    # show rough estimate of how much we're spending.
                    # for equity instruments with margin, we use the margin buy requirement as the cost estimate.
                    # for non-equity (options) without margin, we use the absolute value of the buying power drawdown for the purchase.
                    # TODO: this value is somewhere between wrong or excessive if there's already marginable positions engaged since
                    #       the calculation here is assuming a new position request is the only position in the account.

                    # don't print if this is a failed preview (this max float value is just the ibkr default way of saying "value does not exist")
                    if isset(trade.initMarginAfter):
                        # Logic is a bit weird here because we need to account for various circumstances:
                        #  - equity trades (uses initial margin changes)
                        #  - option trades (uses reduction in equity change)
                        #  - option trades when holding equity positions (margin present, but not used for trade)
                        marginDiff = float(trade.initMarginAfter) - float(
                            trade.initMarginBefore
                        )
                        fundsDiff = marginDiff or abs(float(trade.equityWithLoanChange))
                        logger.info(
                            "[{}] PREVIEW TRADE PERCENTAGE OF AVAILABLE FUNDS: {:,.2f} %",
                            desc,
                            100 * fundsDiff / self.accountStatus["AvailableFunds"],
                        )

            return dict(
                symbol=contract.localSymbol,
                marginInit=margPctInit,
                marginMaint=margPctMaint,
                multiplier=multiplier,
            )

        logger.info("[{}] Ordering {} via {}", desc, contract, order)

        profitTrade = None
        lossTrade = None
        trade = self.ib.placeOrder(contract, order)

        if profitOrder:
            profitTrade = self.ib.placeOrder(contract, profitOrder)

        if lossOrder:
            lossTrade = self.ib.placeOrder(contract, lossOrder)

        # TODO: add optional agent-like feature HERE to modify order in steps for buys (+price, -qty)
        #       or for sells (-price).
        # TODO: move order logic from "buy" lang.py cmd to its own agent feature.
        #       Needs: agent state logged to persistnet data structure, check events on callback for next event in graph (custom OTOCO, etc).
        logger.info(
            "[{} :: {} :: {}] Placed: {}",
            trade.orderStatus.orderId,
            trade.orderStatus.status,
            name,
            pp.pformat(trade),
        )

        if profitOrder:
            logger.info(
                "[{} :: {} :: {}] Profit Order Placed: {}",
                profitTrade.orderStatus.orderId,
                profitTrade.orderStatus.status,
                name,
                pp.pformat(profitTrade),
            )

        if lossOrder:
            logger.info(
                "[{} :: {} :: {}] Loss Order Placed: {}",
                lossTrade.orderStatus.orderId,
                lossTrade.orderStatus.status,
                name,
                pp.pformat(lossTrade),
            )

        return order, trade

    def amountForTrade(self, trade: Trade) -> tuple[float, float, float, float | int]:
        """Return dollar amount of trade given current limit price and quantity.

        Also compensates for contract multipliers correctly.

        Returns:
            - calculated remaining amount
            - calculated total amount
            - current limit price
            - current quantity remaining
        """

        currentPrice = trade.order.lmtPrice
        remainingQty = trade.orderStatus.remaining
        totalQty = remainingQty + trade.orderStatus.filled
        avgFillPrice = trade.orderStatus.avgFillPrice

        # If contract has multiplier (like 100 underlying per option),
        # calculate total spend with mul * p * q.
        # The default "no multiplier" value is '', so this check should be fine.
        if isinstance(trade.contract, Future):
            # FUTURES HACK BECAUSE WE DO EXTERNAL MARGIN CALCULATIONS REGARDLESS OF MULTIPLIER
            mul = 1
        else:
            mul = float(trade.contract.multiplier or 1)

        # use average price IF fills have happened, else use current limit price
        return (
            # Remaining amount to spend
            remainingQty * currentPrice * mul,
            # Total current spend
            totalQty * (avgFillPrice or currentPrice) * mul,
            # current order price limit
            currentPrice,
            # current order remaining amount
            remainingQty,
        )

    def quantityForAmount(
        self, contract: Contract, amount: float, limitPrice: float
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
        if isinstance(contract, Option):
            mul = float(contract.multiplier or 1)
        elif isinstance(contract, Bag):
            # TODO: we should be calculating the bag multipler in a better way probably, but for now we assume all bags have 100 multipliers
            mul = 100
        else:
            mul = 1

        assert mul > 0

        # total spend amount divided by price of thing to buy == how many things to buy
        # (rounding to fix IBKR error for fractional qty: "TotalQuantity field cannot contain more than 8 decimals")
        qty = round(amount / (limitPrice * mul), 8)
        if qty <= 0:
            logger.error(
                "Sorry, your calculated quantity is {:,.8f} so we can't order anything!",
                qty,
            )
            return

        if not isinstance(contract, Crypto):
            # only crypto orders support fractional quantities over the API.
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

    def orderPriceForContract(self, contract: Contract, positionSize: int):
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
                ts.append(
                    (
                        int(t.orderStatus.remaining),
                        float(np.sign(positionSize) * -1 * t.order.lmtPrice),
                    )
                )

        # if only one and it's the full position, return without formatting
        if len(ts) == 1:
            if abs(int(positionSize)) == abs(ts[0][0]):
                return ts[0][1]

        # else, break out by order size, sorted from smallest to largest exit prices
        return sorted(ts, key=lambda x: abs(x[1]))

    def currentQuote(self, sym, show=True) -> tuple[float, float] | None:
        # TODO: maybe we should refactor this to only accept qualified contracts as input (instead of string symbol names) to avoid naming confusion?
        q = self.quoteState.get(sym)
        assert q and q.contract, f"Why doesn't {sym} exist in the quote state?"

        # only optionally print the quote because printing technically requires extra time
        # for all the formatting and display output
        if show:
            ago = (self.now - (q.time or self.now)).as_duration()

            show = [
                f"{q.contract.localSymbol or q.contract.symbol}:",
                f"bid {q.bid:,.2f} x {q.bidSize}",
                f"mid {(q.bid + q.ask) / 2:,.2f}",
                f"ask {q.ask:,.2f} x {q.askSize}",
                f"last {q.last:,.2f} x {q.lastSize}",
                f"ago {str(ago)}",
            ]
            logger.info("    ".join(show))

        # if no quote yet (or no prices available), return last seen price...
        if all(np.isnan([q.bid, q.ask])) or (q.bid <= 0 and q.ask <= 0):
            # for now, disable "last" short circuit reporting because it broke
            # our real time price checks by showing the last closing price of the previous
            # day instead of the "last live trade" as we expected...
            if False:
                if q.last == q.last:
                    return q.last, q.last

            return None

        return q.bid, q.ask

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
                await asyncio.wait_for(self.ib.reqExecutionsAsync(), 3)
        finally:
            # allow the commission report event handler to run again
            self.loadingCommissions = False

    def updatePosition(self, pos):
        self.position[pos.contract.symbol] = pos

    def updateOrder(self, trade):
        self.order[trade.contract.symbol] = trade

        # Only print update if this is regular runtime and not
        # the "load all trades on startup" cycle
        if self.connected:
            logger.warning(
                "[{} :: {} :: {}] Order update: {}",
                trade.orderStatus.orderId,
                trade.orderStatus.status,
                trade.contract.localSymbol,
                trade,
            )

    def errorHandler(self, reqId, errorCode, errorString, contract):
        # Official error code list:
        # https://interactivebrokers.github.io/tws-api/message_codes.html
        if errorCode in {1102, 2104, 2106, 2158, 202}:
            # non-error status codes on startup
            # also we ignore reqId here because it is always -1
            logger.info(
                "API Status {}[code {}]: {}",
                f"[orderId {reqId}] " if reqId else "",
                errorCode,
                errorString,
            )
        else:
            logger.opt(depth=1).error(
                "API Error [orderId {}] [code {}]: {}{}",
                reqId,
                errorCode,
                readableHTML(errorString),
                f" for {contract}" if contract else "",
            )

    def cancelHandler(self, err):
        logger.warning("Order canceled: {}", err)

    def commissionHandler(self, trade, fill, report):
        # Only report commissions if not bulk loading them as a refresh
        # (the bulk load API causes the event handler to fire for each historical fill)
        if self.loadingCommissions:
            logger.warning(
                "[{} :: {} {:>7.2f} of {:>7.2f} :: {}] Ignoring commission because bulk loading history...",
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

        if self.alert:
            if fill.execution.side == "BOT":
                # send BUY note
                ...
            elif fill.execution.side == "SLD":
                # send SOLD note
                ...

        if ICLI_AWWDIO_URL:
            # replace "BOT" and "SLD" with real words because the text-to-speech was pronouncing "SLD" as individual letters "S-L-D"
            side = "bought" if fill.execution.side == "BOT" else "sold"

            fillQty = f"{fill.contract.localSymbol} ({side} {int(fill.execution.shares)} (for {int(fill.execution.cumQty)} of {int(trade.order.totalQuantity)}))"

            #  This triggers on a successful close of a position (TODO: need to fill out more details)
            if fill.commissionReport.realizedPNL:
                PorL = "profit" if fill.commissionReport.realizedPNL >= 0 else "loss"

                asyncio.create_task(
                    self.speak.say(
                        say=f"CLOSED: {trade.orderStatus.status} FOR {fillQty} ({PorL} ${round(fill.commissionReport.realizedPNL, 2):,})"
                    )
                )
            else:
                # We notify about orders HERE instead of in 'orderExecuteHandler()' because HERE we have details about filled/canceled for
                # the status, where 'orderExecuteHandler()' always just has status of "Submitted" when an execution happens (also with no price details) which isn't as useful.
                asyncio.create_task(
                    self.speak.say(
                        say=f"OPENED: {trade.orderStatus.status} FOR {fillQty} (commission {locale.currency(fill.commissionReport.commission)})"
                    )
                )

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
            locale.currency(fill.commissionReport.commission),
            locale.currency(fill.commissionReport.commission / fill.execution.shares),
            f" (pnl {fill.commissionReport.realizedPNL:,.2f})"
            if fill.commissionReport.realizedPNL
            else "",
        )

    def newsBHandler(self, news: NewsBulletin):
        logger.warning("News Bulletin: {}", readableHTML(news.message))

    def newsTHandler(self, news: NewsTick):
        logger.warning("News Tick: {}", news)

    def orderExecuteHandler(self, trade, fill):
        logger.warning(
            "[{} :: {} :: {}] Trade executed for {}",
            trade.orderStatus.orderId,
            trade.orderStatus.status,
            trade.contract.localSymbol,
            fill.contract.localSymbol,
        )

        if fill.execution.cumQty > 0:
            if trade.contract.conId not in self.pnlSingle:
                self.pnlSingle[trade.contract.conId] = self.ib.reqPnLSingle(
                    self.accountId, "", trade.contract.conId
                )
        else:
            # if quantity is gone, stop listening for updates and remove.
            self.ib.cancelPnLSingle(self.accountId, "", trade.contract.conId)
            del self.pnlSingle[trade.contract.conId]

    def tickersUpdate(self, tickr):
        """This runs on EVERY quote update which happens 4 times per second per subsubscribed symbol.

        We don't technically need this to receive ticker updates since tickers are "live updated" in their
        own classes for reading, but we _do_ use this to calculate live metadata, reporting, or quote-based
        algo triggers.

        This method should always be clean and fast because it runs up to 100+ times per second depending on how
        many tickers you are subscribed to in your client.

        Also note: because this is an ib_insync event handler, any errors or exceptions in this method are NOT
                   reported to the main program. You should attach @logger.catch to this method if you think it
                   isn't working correctly because then you can see the errors/exceptions (if any).
        """
        # logger.info("Ticker update: {}", tickr)

        for ticker in tickr:
            c = ticker.contract
            name = (c.localSymbol or c.symbol).replace(" ", "")
            price = (ticker.bid + ticker.ask) / 2

            # "Bag" spreads have no unique ID themselves and Contract objects can't be hashed because
            # they are mutable, so we generate a fast synthetic bag quote key here (which is mostly like
            # the global quote key except we aren't bothering to sort it here... the sort probably isn't necessary
            # in the global one either)
            if isinstance(c, Bag):
                quotekey = tuple(x.tuple() for x in c.comboLegs)
            else:
                quotekey = c.conId

            self.quotehistory[quotekey].append(price)

            # this is a synthetic memory-having ATR where we just feed it price data and
            # it calculates a dynamic H/L/C for the actual ATR based on recent price history.
            if ticker.bid > 0 and ticker.ask > 0:
                self.atrs[name].update(price)

                if False and ICLI_AWWDIO_URL:
                    asyncio.create_task(
                        self.speak.say(say=f"PRICE UP: {name} TO {price:,.2f}")
                    )

        # TODO: we could also run volume crossover calculations too...

        # TODO: we should also do some algo checks here based on the live quote price updates...

        if ICLI_DUMP_QUOTES:
            with open(
                f"tickers-{datetime.datetime.now().date()}-{ICLI_CLIENT_ID}.json", "a"
            ) as tj:
                for ticker in tickr:
                    tj.write(
                        orjson.dumps(
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
                    tj.write("\n")

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

        # collect updates into a single update dict so we can re-broadcast this update
        # to external agent listeners too all at once.
        update = {}
        if v.tag in STATUS_FIELDS_PROCESS:
            try:
                match v.tag:
                    case "BuyingPower":
                        # regular 25% margin for boring symbols
                        update["BuyingPower4"] = float(v.value)

                        # 30% margin for "regular" symbols
                        update["BuyingPower3"] = float(v.value) / 1.3333333333

                        # 50% margin for overnight or "really exciting" symbols
                        update["BuyingPower2"] = float(v.value) / 2
                    case "NetLiquidation":
                        nl = float(v.value)
                        update[v.tag] = nl
                        upl = self.accountStatus.get("UnrealizedPnL", 0)
                        rpl = self.accountStatus.get("RealizedPnL", 0)

                        # Also generate some synthetic data about percentage gains we made.
                        # Is this accurate enough? Should we be doing the math differently or basing it off AvailableFunds or BuyingPower instead???
                        # We subtract the PnL values from the account NetLiquidation because the PnL contribution is *already* accounted for
                        # in the NetLiquididation value.
                        # (the updates are *here* because this runs on every NetLiq val update instead of ONLY on P&L updates)
                        update["RealizedPnL%"] = (rpl / (nl - rpl)) * 100
                        update["UnrealizedPnL%"] = (upl / (nl - upl)) * 100

                        # Also combine realized+unrealized to show the current daily total PnL percentage because
                        # maybe we have 12% realized profit but -12% unrealized and we're actually flat...
                        update["DayPnL%"] = (
                            update["RealizedPnL%"] + update["UnrealizedPnL%"]
                        )
                    case _:
                        self.accountStatus[v.tag] = float(v.value)
            except:
                # don't care, just keep going
                pass
            finally:
                self.accountStatus |= update
                self.updateAgentAccountStatus("summary", update)

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

    def bottomToolbar(self):
        self.updates += 1
        self.now = pendulum.now("US/Eastern")

        def fmtPrice2(n: float):
            # Some prices may not be populated if they haven't
            # happened yet (e.g. PNL values if no trades for the day yet, etc)
            if not n:
                n = 0

            # if GTE $1 million, stop showing cents.
            if n > 999_999.99:
                return f"{n:>10,.0f}"

            return f"{n:>10,.2f}"

        def fmtEquitySpread(n):
            if isinstance(n, (int, float)):
                return f"{n:>6,.2f}"

            return f"{n:>5}"

        def fmtPriceOpt(n):
            if isinstance(n, (int, float)):
                # assume trading $0.01 to $99.99 range for options
                # (we can get integers here if we decided there's no valid bid
                #  and we're just marking a price to 0)
                return f"{n:>5,.2f}"

            return f"{n:>5}"

        def updateEMA(sym, price, longOnly=True):
            # if no price, don't update (but allow negative prices if this is a credit quote)
            if (not price) or (longOnly and (price < 0)) or (price != price):
                # logger.info("Skipping EMA update for: {} because {}", sym, price)
                return

            # Normalize the EMAs s so they are in TIME and not "updates per ICLI_REFRESH interval"
            # 1 minute and 3 minute EMAs

            # these are in units of fractional seconds we need to normalize to our "bar update duration intervals"
            # TODO: we could move this to the ticker updater instead? then it updates on every quote change and we have timestamps to use.
            refresh = self.toolbarUpdateInterval

            MIN_1 = 60 // refresh
            MIN_3 = (60 * 3) // refresh

            for name, back in (("1m", MIN_1), ("3m", MIN_3)):
                prev = self.ema[sym][name]

                # use previous price or initialize with current price
                if (not prev) or (prev != prev):
                    prev = price

                # fmt: off
                # if prev != prev:
                #    logger.info("NaN in EMA? [{} :: {}] {} {} -> {}", prev, price, sym, back, self.ema[sym][back])
                # fmt: on

                k = 2 / (back + 1)
                self.ema[sym][name] = (k * (price - prev)) + prev

        def getEMA(sym, name, roundto=2):
            # Round our results here so we don't need to excessively format all the prints.
            return round(self.ema[sym][name], roundto)

        # Fields described at:
        # https://ib-insync.readthedocs.io/api.html#module-ib_insync.ticker
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
            bid = c.bid
            ask = c.ask
            bidSize = c.bidSize
            askSize = c.askSize
            decimals = 2

            if isinstance(c.contract, Forex):
                decimals = 5

            if bid > 0 and bid == bid and ask > 0 and ask == ask:
                if isinstance(c.contract, Future):
                    usePrice = rounder.round("/" + c.contract.symbol, (bid + ask) / 2)
                else:
                    usePrice = round((bid + ask) / 2, decimals)
            elif isinstance(c.contract, Bag):
                if (bid != 0 and ask != 0) and (bid == bid) and (ask == ask):
                    # bags are allowed to have negative prices because they can be credit quotes
                    usePrice = round((bid + ask) / 2, 2)
                else:
                    # else, IBKR isn't giving us quotes for this spread (IBKR datafeeds suck), so actually let's
                    # see if we have a live quote running for all legs so we can generate the spread quote
                    # anyway with our own math... sigh.
                    # logger.info("Generating synthetic quote using: {}", c.contract)
                    contractIds = [x.conId for x in c.contract.comboLegs]
                    quotes = [
                        # .get() because MAYBE THIS QUOTE DOESN'T EXIST EITHER, so we can't even generate a synthetic bag quote. sad.
                        self.quoteState.get(quotekey)
                        for quotekey in [
                            self.contractIdsToQuoteKeysMappings.get(x)
                            for x in contractIds
                        ]
                    ]
                    # logger.info("Found underlying quotes: {}", quotes)
                    bid = 0
                    ask = 0
                    bidSize = float("inf")
                    askSize = float("inf")
                    for leg, quote in zip(c.contract.comboLegs, quotes):
                        # if a quote doesn't exist, we need to abandon trying to generate any part of this synthetic quote
                        # because we don't have all the data we need so just combining partial values would be wrong.
                        if not quote or (quote.ask == -1 or quote.bid == -1):
                            bid = 0
                            ask = 0
                            bidSize = 0
                            askSize = 0
                            usePrice = np.nan
                            break

                        if leg.action == "SELL":
                            # SELL legs have opposite signs and positions because they are credits
                            bid -= quote.ask * leg.ratio
                            ask -= quote.bid * leg.ratio
                            # the "quantity" of a spread is the smallest number available for the combinations
                            bidSize = min(bidSize, quote.askSize)
                            askSize = min(askSize, quote.bidSize)
                        else:
                            # else, is BUY, so we do normal adding to the normal order
                            bid += quote.bid * leg.ratio
                            ask += quote.ask * leg.ratio
                            bidSize = min(bidSize, quote.bidSize)
                            askSize = min(askSize, quote.askSize)
                    else:
                        usePrice = round((bid + ask) / 2, 2)
            else:
                # else, bid/ask is currently offline or broken or just not on the symbol, so use the last traded price.
                usePrice = c.last if c.last == c.last else c.close

            # update EMA using current midpoint estimate (and allow Bag/spread quotes to have negative EMA due to credit spreads)
            updateEMA(ls, usePrice, not isinstance(c.contract, Bag))

            ago = (self.now - (c.time or self.now)).as_duration()
            try:
                percentUnderHigh = (
                    ((usePrice - c.high) / c.high) * 100 if c.high == c.high else 0
                )

                percentUpFromLow = (
                    ((usePrice - c.low) / c.low) * 100 if c.low == c.low else 0
                )

                percentUpFromClose = (
                    ((usePrice - c.close) / c.close) * 100 if c.close else 0
                )
            except:
                # price + (low or close) is zero... can't do that.
                percentUpFromLow = 0
                percentUpFromClose = 0

            def mkcolor(
                n: float, vals: str | list[str], colorRanges: list[str]
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

            amtHigh = usePrice - c.high
            amtLow = usePrice - c.low
            amtClose = usePrice - c.close
            # If there are > 1,000 point swings, stop displaying cents.
            # also the point differences use the same colors as the percent differences
            # because having fixed point color offsets doesn't make sense (e.g. AAPL moves $2
            # vs DIA moving $200)

            # if bidsize or asksize are > 100,000, just show "100k" instead of breaking
            # the interface for being too wide

            if np.isnan(bidSize) or np.isinf(bidSize):
                b_s = f"{'X':>6}"
            elif 0 < bidSize < 1:
                # there's a bug here when 'bidSize' is 'inf' and it's triggering here??
                b_s = f"{bidSize:>6.4f}"
            elif bidSize < 100_000:
                b_s = f"{int(bidSize):>6,}"
            else:
                b_s = f"{int(bidSize // 1000):>5,}k"

            if np.isnan(askSize) or np.isinf(askSize):
                a_s = f"{'X':>6}"
            elif 0 < askSize < 1:
                a_s = f"{askSize:>6.4f}"
            elif askSize < 100_000 or np.isnan(askSize):
                a_s = f"{int(askSize):>6,}"
            else:
                a_s = f"{int(askSize // 1000):>5,}k"

            # use different print logic if this is an option contract or spread
            bigboi = isinstance(c.contract, (Option, FuturesOption, Bag))

            if bigboi:
                # Could use this too, but it only updates every couple seconds instead
                # of truly live with each new bid/ask update.
                # if c.modelGreeks:
                #     mark = c.modelGreeks.optPrice

                if bid and bidSize and ask and askSize:
                    mark = round((bid + ask) / 2, 2)
                    # weighted sum of bid/ask as midpoint
                    # We do extra rounding here so we don't end up with
                    # something like "$0.015" when we really want "$0.01"
                    # mark = round(
                    #     ((c.bid * c.bidSize) + (c.ask * c.askSize))
                    #     / (c.bidSize + c.askSize),
                    #     2,
                    # )
                else:
                    # IBKR reports "no bid" as -1 but when bid is -1 bidSize is 0.
                    # If no bid, there's no valid midpoint so just go to the ask directly.
                    # Different views though: for BUYING, the price is the ask with no midpoint,
                    #                         for SELLING, the price DOES NOT EXIST because no buyers.
                    mark = round((bid + ask) / 2, 2) if bid > 0 else 0

                e100 = getEMA(ls, "1m")
                e300 = getEMA(ls, "3m")
                # logger.info("[{}] Got EMA for OPT: {} -> {}", ls, e100, e300)
                e100diff = (mark - e100) if e100 else None

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
                bighigh = ((mark / c.high if c.high else 1) - 1) * 100

                # only report low if current mark estimate is ABOVE the registered
                # low for the day, else we report it as currently trading AT the low
                # for the day instead of potentially BELOW the low for the day.
                biglow = ((mark / c.low if c.low else 1) - 1) * 100

                bigclose = ((mark / c.close if c.close else 1) - 1) * 100

                pctBigHigh, amtBigHigh = mkPctColor(
                    bighigh,
                    [
                        f"{bighigh:>7.2f}%",
                        f"{amtHigh:>7.2f}" if amtHigh < 1000 else f"{amtHigh:>7.0f}",
                    ],
                )
                pctBigLow, amtBigLow = mkPctColor(
                    biglow,
                    [
                        f"{biglow:>7.2f}%",
                        f"{amtLow:>7.2f}" if amtLow < 1000 else f"{amtLow:>7.0f}",
                    ],
                )
                pctBigClose, amtBigClose = mkPctColor(
                    bigclose,
                    [
                        f"{bigclose:>7.2f}%",
                        f"{amtClose:>7.2f}" if amtLow < 1000 else f"{amtClose:>7.0f}",
                    ],
                )

                if False:
                    pctUpLow, amtUpLow = mkPctColor(
                        percentUpFromLow,
                        [
                            f"{percentUpFromLow:>7.2f}%",
                            f"{amtLow:>7.2f}" if amtLow < 1000 else f"{amtLow:>7.0f}",
                        ],
                    )
                    pctUpClose, amtUpClose = mkPctColor(
                        percentUpFromClose,
                        [
                            f"{percentUpFromClose:>7.2f}%",
                            f"{amtClose:>7.2f}"
                            if amtLow < 1000
                            else f"{amtClose:>7.0f}",
                        ],
                    )

                if c.lastGreeks and c.lastGreeks.undPrice:
                    und = c.lastGreeks.undPrice
                    strike = c.contract.strike
                    underlyingStrikeDifference = -(strike - und) / und * 100
                    iv = c.lastGreeks.impliedVol
                    # for our buying and selling, we want greeks based on the live floating
                    # bid/ask spread and not the last price (could be out of date) and not
                    # the direct bid or ask (too biased while buying and selling)
                    delta = c.modelGreeks.delta if c.modelGreeks else None
                else:
                    und = None
                    underlyingStrikeDifference = None
                    iv = None
                    delta = None

                # Note: we omit OPEN price because IBKR doesn't report it (for some reason?)
                # greeks available as .bidGreeks, .askGreeks, .lastGreeks, .modelGreeks each as an OptionComputation named tuple
                # '.halted' is either nan or 0 if NOT halted, so 'halted > 0' should be a safe check.
                rowName: str

                # For all combos, we cache the ID to original symbol mapping
                # after the contractId is resolved.
                if c.contract.comboLegs:
                    # generate rows to look like:
                    # B  1 AAPL212121C000...
                    # S  2 ....
                    rns = []
                    for idx, x in enumerate(c.contract.comboLegs):
                        contract = self.conIdCache[x.conId]
                        padding = "    " if idx > 0 else ""
                        rns.append(
                            f"{padding}{x.action[0]} {x.ratio:2} {contract.localSymbol or contract.symbol}"
                        )

                    if False:
                        logger.info(
                            "Contract and vals for combo: {}  -> {} -> {} -> {} -> {}",
                            c.contract,
                            ls,
                            e100,
                            e300,
                            (usePrice, c.bid, c.ask, c.high, c.low),
                        )

                    rowName = "\n".join(rns)

                    # show a recent range of prices since spreads have twice (or more) the bid/ask volatility
                    # of a single leg option (due to all the legs being combined into one quote dynamically)
                    src = self.quotehistory[
                        tuple(x.tuple() for x in c.contract.comboLegs)
                    ]

                    # typically, a low stddev indicates temporary low volatility which is
                    # the calm before the storm when a big move happens next (in either direction,
                    # but direction prediction can be augmented with moving average crossovers).
                    try:
                        std = statistics.stdev(src)
                    except:
                        std = 0

                    try:
                        parts = [
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

                    parts = ", ".join(
                        [
                            f"{x:>7.2f}" if isinstance(x, (float, int)) else x
                            for x in parts
                        ]
                    )

                    # Some of the daily values seem to exist for spreads: high and low of day, but previous day close just reports the current price.
                    return " ".join(
                        [
                            rowName,
                            f"{fmtPriceOpt(e100):>6}",
                            f"{trend}",
                            f"{fmtPriceOpt(e300):>6}",
                            f"   {fmtPriceOpt(mark):>6} ±{fmtPriceOpt(ask - mark):<4}",
                            f" ({pctBigHigh} {amtBigHigh} {fmtPriceOpt(c.high):>6})",
                            f"({pctBigLow} {amtBigLow} {fmtPriceOpt(c.low):>6})",
                            f" {fmtPriceOpt(bid):>6} x {b_s}   {fmtPriceOpt(ask):>6} x {a_s} ",
                            f"  ({str(ago):>13})  ",
                            f"  :: {parts}  (r {minmax:.2f})  (s {std:.2f})",
                            "HALTED!" if c.halted > 0 else "",
                        ]
                    )
                else:
                    rowName = f"{c.contract.localSymbol or c.contract.symbol:<21}:"

                    try:
                        contract = c.contract
                        if isinstance(contract, (Option, FuturesOption)):
                            # has data like:
                            # FuturesOption(conId=653770578, symbol='RTY', lastTradeDateOrContractMonth='20231117', strike=1775.0, right='P', multiplier='50', exchange='CME', currency='USD', localSymbol='R3EX3 P1775', tradingClass='R3E')
                            ltdocm = contract.lastTradeDateOrContractMonth
                            y = ltdocm[2:4]
                            m = ltdocm[4:6]
                            d = ltdocm[6:8]
                            pc = contract.right
                            price = contract.strike
                            sym = rowName
                            rowNice = f"{sym} {y}-{m}-{d} {pc} {price:>8,.2f}"
                    except:
                        # else, we can't parse it for some reason, so juse use the name...
                        rowNice = rowName

                    # TODO: should this be fancier and decay cleaner?
                    #       we could do more accurate countdowns to actual expiration time instead of just "days"
                    # TODO: we could use market calendars to generate the proper instrument stop time to account for
                    #       early closes the 1-2 days per year when those happen.
                    when = (
                        pendulum.parse(f"20{y}-{m}-{d} 16:00", tz="US/Eastern")
                        - self.now
                    ).days

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
                    compdiff = 0
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

                    return " ".join(
                        [
                            rowName,
                            f"[u {fmtPricePad(und, padding=8, decimals=2)} ({itm:<1} {underlyingStrikeDifference or -0:>7,.2f}%)]",
                            f"[iv {iv or 0:.2f}]",
                            f"[d {delta or 0:>5.2f}]",
                            f"{fmtPriceOpt(e100):>6}",
                            f"{trend}",
                            f"{fmtPriceOpt(e300):>6}",
                            f"{fmtPriceOpt(mark):>6} ±{fmtPriceOpt(c.ask - mark):<4}",
                            f"({pctBigHigh} {amtBigHigh} {fmtPriceOpt(c.high):>6})",
                            f"({pctBigLow} {amtBigLow} {fmtPriceOpt(c.low):>6})",
                            f"({pctBigClose} {amtBigClose} {fmtPriceOpt(c.close):>6})",
                            f" {fmtPriceOpt(c.bid):>6} x {b_s}   {fmtPriceOpt(c.ask):>6} x {a_s} ",
                            f"  ({str(ago):>13})  ",
                            f"(s {fmtPricePad(compensated, padding=8, decimals=2)} @ {compdiff:>6,.2f})",
                            rowNice,
                            f"({when:>3} d)",
                            "HALTED!" if c.halted > 0 else "",
                        ]
                    )

            # TODO: pre-market and after-market hours don't update the high/low values, so these are
            #       not populated during those sessions.
            #       this also means during after-hours session, the high and low are fixed to what they
            #       were during RTH and are no longer valid. Should this have a time check too?
            # TODO: replace these fixed 6.2 and 8.2 formats with fmtPricePad() with proper decimal extension for forex values instead.
            pctUndHigh, amtUndHigh = mkPctColor(
                percentUnderHigh,
                [
                    f"{percentUnderHigh:>6.2f}%",
                    f"{amtHigh:>8.2f}" if amtHigh < 1000 else f"{amtHigh:>8.0f}",
                ],
            )
            pctUpLow, amtUpLow = mkPctColor(
                percentUpFromLow,
                [
                    f"{percentUpFromLow:>5.2f}%",
                    f"{amtLow:>6.2f}" if amtLow < 1000 else f"{amtLow:>6.0f}",
                ],
            )

            # high and low are only populated after regular market hours, so allow nan to show the
            # full float value during pre-market hours.
            pctUpClose, amtUpClose = mkPctColor(
                percentUpFromClose,
                [
                    f"{percentUpFromClose:>6.2f}%",
                    f"{amtClose:>8.2f}"
                    if (amtLow != amtLow) or amtLow < 1000
                    else f"{amtClose:>8.0f}",
                ],
            )

            # somewhat circuitous logic to format NaNs and values properly at the same string padding offsets
            atrval = np.nan
            if atrr := self.atrs.get(ls):
                atrval = atrr.atr.current

            atr = f"{atrval:>5.2f}"

            roundto = 2
            # symbol exceptions for things we want bigger (GBP is a future and not a Forex...)
            # TODO: fix for 3-decimal futures too.
            if ls in {"GBP"}:
                decimals = 4

            e100 = getEMA(ls, "1m", decimals)
            e300 = getEMA(ls, "3m", decimals)

            # for price differences we show the difference as if holding a LONG position
            # at the historical price as compared against the current price.
            # (so, if e100 is $50 but current price is $55, our difference is +5 because
            #      we'd have a +5 profit if held from the historical price.
            #      This helps align "price think" instead of showing difference from historical
            #      vs. current where "smaller historical vs. larger current" would cause negative
            #      difference which is actually a profit if it were LONG'd in the past)
            # also don't show differences for TICK because it's not really a useful number (and it's too big breaking formatting)
            if ls == "TICK-NYSE":
                e100diff = np.nan
                e300diff = np.nan
            else:
                e100diff = (usePrice - e100) if e100 else None
                e300diff = (usePrice - e300) if e300 else None
            # logger.info("[{}] e100 e300: {} {} {} {}", ls, e100, e300, e100diff, e300diff)

            # also add a marker for if the short term trend (1m) is GT, LT, or EQ to the longer term trend (3m)
            ediff = e100 - e300
            if ediff > 0:
                trend = "&gt;"
            elif ediff < 0:
                trend = "&lt;"
            else:
                trend = "="

            return " ".join(
                [
                    f"{ls:<9}",
                    f"{fmtPricePad(e100, decimals=decimals)}",
                    f"({fmtPricePad(e100diff, padding=6, decimals=3)})",
                    f"{trend}",
                    f"{fmtPricePad(e300, decimals=decimals)}",
                    f"({fmtPricePad(e300diff, padding=6, decimals=3)})",
                    f"{fmtPricePad(usePrice, decimals=decimals)} ±{fmtEquitySpread(c.ask - usePrice) if c.ask >= usePrice else '':<6}",
                    f"({pctUndHigh} {amtUndHigh})",
                    f"({pctUpLow} {amtUpLow})",
                    f"({pctUpClose} {amtUpClose})",
                    f"{fmtPricePad(c.high, decimals=decimals)}",
                    f"{fmtPricePad(c.low, decimals=decimals)}",
                    f"<aaa bg='purple'>{fmtPricePad(c.bid, decimals=decimals)} x {b_s} {fmtPricePad(c.ask, decimals=decimals)} x {a_s}</aaa>",
                    f"({atr})",
                    f"{fmtPricePad(c.open, decimals=decimals)}",
                    f"{fmtPricePad(c.close, decimals=decimals)}",
                    f"({str(ago)})",
                    "     HALTED!" if c.halted > 0 else "",
                ]
            )

        try:
            rowlen, _ = shutil.get_terminal_size()

            rowvals = [[]]
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
                if (currentrowlen + vlen + 4) < rowlen:
                    # append to current row
                    rowvals[-1].append(value)
                    currentrowlen += vlen + 4
                else:
                    # add new row, reset row length
                    rowvals.append([value])
                    currentrowlen = vlen

            balrows = "\n".join("    ".join(x) for x in rowvals)

            def sortQuotes(x):
                """Comparison function to sort quotes by specific types we want grouped together."""
                sym, quote = x
                c = quote.contract

                # We want to sort futures first, and sort MES, MNQ, etc first.
                # (also Indexes and Index ETFs first too)
                # This double symbol check is so we don't accidentially sort market ETF options
                # inside the regular equity section.
                if c.secType in {"FUT", "IND"} or (
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
                    return (0, priority, c.secType, c.symbol)

                # draw crypto and forex/cash quotes under futures quotes
                if c.secType in {"CRYPTO", "CASH"}:
                    priority = 0
                    return (0, priority, c.secType, c.symbol)

                if c.secType == "OPT":
                    # options are medium last because they are wide
                    priority = 0
                    return (2, priority, c.secType, c.localSymbol)

                if c.secType == "FOP":
                    # future options are above other options...
                    priority = -1
                    return (2, priority, c.secType, c.localSymbol)

                if c.secType == "BAG":
                    # bags are last because their descriptions are big
                    priority = 0
                    return (3, priority, c.secType, c.symbol)

                # else, just by name.
                # BUT we do these in REVERSE order since they
                # are at the end of the table!
                # (We create "reverse order" by translating all
                #  letters into their "inverse" where a == z, b == y, etc).
                priority = 0
                return (1, priority, c.secType, invertstr(c.symbol.lower()))

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

            qs = sorted(self.quoteState.items(), key=sortQuotes)
            self.quotesPositional = qs

            spxbreakers = ""
            spx = self.quoteState.get("SPX")
            if spx:
                # hack around IBKR quotes being broken over weekends/holdays
                # NOTE: this isn't valid across weekends because until Monday morning, the "close" is "Thursday close" not frday close. sigh.
                #       also the SPX symbol never has '.open' value so we can't detect "stale vs. current quote from last close"
                spxc = spx.close
                spxl = spx.last

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

            # TODO: we may want to iterate these to exclude "Inactive" orders like:
            # [x.log[-1].status == "Inactive" for x in self.ib.openTrades()]
            ordcount = len(self.ib.openTrades())
            openorders = f"open orders: {ordcount:,}"

            positioncount = len(self.ib.portfolio())
            openpositions = f"positions: {positioncount:,}"

            executioncount = len(self.ib.fills())
            todayexecutions = f"executions: {executioncount:,}"

            # TODO: We couold also flip this between a "time until market open" vs "time until close" value depending
            #       on if we are out of market hours or not, but we aren't bothering with the extra logic for now.
            untilClose = fetchEndOfMarketDay() - self.now
            todayclose = f"mktclose: {untilClose.in_words()}"

            return HTML(
                # all these spaces look weird, but they (kinda) match the underlying column-based formatting offsets
                f"""[{ICLI_CLIENT_ID}] {self.now}{onc} [{self.updates:,}]                {spxbreakers}                     {openorders}    {openpositions}    {todayexecutions}      {todayclose}\n"""
                + "\n".join(
                    [
                        f"{qp:>2}) " + formatTicker(quote)
                        for qp, (sym, quote) in enumerate(qs)
                    ]
                )
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
        )
        for t in terms:
            if not t:
                continue

            got = await t.ask(**extraArgs)

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
            assert contract.conId, f"Sorry, we only accept qualified contracts for adding quotes, but we got: {contract}"

        tickFields = tickFieldsForContract(contract)

        # remove spaces from OCC-like symbols for consistent key reference
        symkey = lookupKey(contract)

        # don't double-subscribe to symbols! If something is already in our quote state, we have an active subscription!
        if symkey not in self.quoteState:
            # logger.info("[{}] Adding new live quote...", symkey)
            self.quoteState[symkey] = self.ib.reqMktData(contract, tickFields)
            self.contractIdsToQuoteKeysMappings[contract.conId] = symkey

            # This is a nice debug helper just showing the quote key name to the attached contract subscription:
            # logger.info("[{}]: {}", symkey, contract)

        return symkey

    def quoteExists(self, contract):
        return lookupKey(contract) in self.quoteState

    async def addQuotes(self, symbols):
        """Add quotes by a common symbol name"""
        if not symbols:
            return

        ors: list[buylang.OrderRequest] = []
        for sym in symbols:
            sym = sym.upper()
            # don't attempt to double subscribe
            # TODO: this only checks the named entry, so we need to verify we aren't double subscribing /ES /ESZ3 etc
            if sym in self.quoteState:
                continue

            # if this is a spread quote, attempt to replace any :N requests with the actual symbols...
            if " " in sym:
                rebuild: list[str] = []
                for part in sym.split():
                    if part[0] == ":":
                        foundSymbol, _contract = self.quoteResolve(part)
                        rebuild.append(foundSymbol)
                    else:
                        rebuild.append(part)

                # now put it back together again...
                sym = " ".join(rebuild)

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

        qs = set()
        for ordReq, contract in zip(ors, cs):
            if not contract:
                logger.error(
                    "Failed to find live contract for: {} :: {}", ordReq, contract
                )
                continue

            symkey = self.addQuoteFromContract(contract)
            qs.add(symkey)

        # return array of quote lookup keys
        # (because things like spreads have weird keys, we construct parts the caller
        #  can then use to index into the quoteState[] dict directly later)
        return list(qs)

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
                if "token" in str(e):
                    # don't show a 100 line stack trace for mistyped inputs.
                    # Just tell the user it needs to be corrected.
                    logger.error("Error parsing your input: {}", e)
                else:
                    logger.exception("sorry, what now?")

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

        runnables = []

        # 'collective' holds the current accumulating concurrency group
        collective = []

        commentRemoved = re.sub(r"#.*", "", text1).strip()
        ccmds = re.split(r"[\n;]", commentRemoved)
        for ccmd in ccmds:
            # if the split generated empty entries (like running ;;;;), just skip the command
            ccmd = ccmd.strip()

            if not ccmd:
                continue

            # custom usability hack: we can detect math ops and not need to prefix 'math' to them manually
            if ccmd[0] == "(":
                ccmd = f"math {ccmd}"

            # Check if this command is a background command then clean it up
            isBackgroundCmd = ccmd[-1] == "&"
            if isBackgroundCmd:
                # remove ampersand from background request and re-strip command...
                ccmd = ccmd[:-1].rstrip()

            # split into command dispatch lookup and arguments to command
            cmd, *rest = ccmd.split(" ", 1)

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
        await self.prepare()
        while not self.exiting:
            try:
                await self.dorepl()
            except:
                logger.exception("Uncaught exception in repl? Restarting...")
                continue

    async def prepare(self):
        # Setup...

        # wait until we start getting data from the gateway...
        loop = asyncio.get_event_loop()

        self.dispatch = lang.Dispatch()

        # flip to enable/disable verbose ib_insync library logging
        if False:
            import logging

            ib_async.util.logToConsole(logging.INFO)

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

        # Note: "PortfolioEvent" is fine here since we are using a single account.
        # If you have multiple accounts, you want positionEvent (the IBKR API
        # doesn't allow "Portfolio" to span accounts, but Positions can be reported
        # from multiple accounts with one API connection apparently)
        self.ib.updatePortfolioEvent += lambda row: self.updatePosition(row)

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
                await self.dispatch.runop("qloadsnapshot", "", self.opstate)

            contracts: list[Stock | Future | Index] = [
                Stock(sym, "SMART", "USD") for sym in stocks
            ]
            contracts += futures
            contracts += idxs

            with Timer("[quotes :: global] Restored quote state"):
                # run restore and local contracts qualification concurrently
                # logger.info("pre=qualified: {}", contracts)
                _, contracts = await asyncio.gather(
                    # restore SHARED global symbols
                    self.dispatch.runop("qrestore", "global", self.opstate),
                    # prepare to restore COMMON symbols
                    self.qualify(*contracts),
                )
                # logger.info("post=qualified: {}", contracts)

            with Timer("[quotes :: common] Restored quote state"):
                for contract in contracts:
                    try:
                        # logger.info("Adding quote for: {} via {}", contract, contracts)
                        self.addQuoteFromContract(contract)
                    except:
                        logger.error("Failed to add on startup: {}", contract)

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
                    self.position.clear()
                    self.order.clear()
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
                        "Connected! Current Request ID for Client {}: {}",
                        self.clientId,
                        self.ib.client._reqIdSeq,
                    )

                    self.connected = True

                    self.ib.reqNewsBulletins(True)

                    # we load executions fully async after the connection happens because
                    # the fetching during connection causes an extra delay we don't need.
                    asyncio.create_task(self.loadExecutions())

                    # also load market data async for quicker non-blocking startup
                    asyncio.create_task(requestMarketData())

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
                        "[{}] Failed to connect to IB Gateway, trying again...", e
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
            await reconnect()
        except SystemExit:
            # do not pass go, do not continue, throw the exit upward
            sys.exit(0)

        set_title(f"{self.levelName().title()} Trader ({self.clientId})")
        self.ib.disconnectedEvent += lambda: asyncio.create_task(reconnect())

    async def dorepl(self):
        session = PromptSession(
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
                    # refresh_interval=3,
                    # mouse_support=True,
                    # completer=completer, # <-- causes not to be full screen due to additional dropdown space
                    complete_in_thread=True,
                    complete_while_typing=True,
                    search_ignore_case=True,
                )

                # log user input to our active logfile(s)
                logger.trace("{}> {}", self.levelName(), text1)

                # 'runnables' is the list of all commands to run after we collect them
                runnables = self.buildRunnablesFromCommandRequest(text1)

                # if no commands, just draw the prompt again
                if not runnables:
                    continue

                if len(runnables) == 1:
                    # if only one command, don't run with an extra Timer() report like we do below
                    # with multiple commands (individual commands always report their individual timing)
                    await runnables[0]
                else:
                    # only show the "All commands" timer if we have multiple commands to run
                    with Timer("All commands"):
                        for run in runnables:
                            try:
                                # run a COLLECTIVE COMMAND GROUP we previously created
                                await run
                            except:
                                logger.exception("[{}] Runnable failed?", run)
            except KeyboardInterrupt:
                # Control-C pressed. Try again.
                continue
            except EOFError:
                # Control-D pressed
                logger.error("Exiting...")
                self.exiting = True
                break
            except BlockingIOError as bioe:
                # this is noisy macOS problem if using a non-fixed
                # uvloop and we don't care, but it will truncate or
                # duplicate your output.
                # solution: don't use uvloop or use a working uvloop
                try:
                    logger.error("FINAL\n")
                except:
                    pass
            except Exception as err:
                while True:
                    try:
                        logger.exception("Trying...")
                        break
                    except Exception as e2:
                        await asyncio.sleep(1)
                        pass

    def stop(self):
        self.ib.disconnect()

    async def setup(self):
        pass
