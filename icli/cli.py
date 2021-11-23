#!/usr/bin/env python3

original_print = print
from prompt_toolkit import print_formatted_text, Application

# from prompt_toolkit import print_formatted_text as print
from prompt_toolkit.formatted_text import HTML

import pathlib
from bs4 import BeautifulSoup

# http://www.grantjenks.com/docs/diskcache/
import diskcache

from icli.futsexchanges import FUTS_EXCHANGE
import decimal
import sys

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import datetime
import os

from typing import Union, Optional, Sequence, Any, Mapping

import numpy as np

import pendulum

import pandas as pd

# for automatic money formatting in some places
import locale

locale.setlocale(locale.LC_ALL, "")

import os

# Tell pygame to not print a hello message when it is imported
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# sounds!
import pygame

import ib_insync
from ib_insync import (
    IB,
    Contract,
    Bag,
    ComboLeg,
    Ticker,
    RealTimeBarList,
    PnLSingle,
    Order,
    NewsBulletin,
    NewsTick,
)
import pprint
import asyncio

import logging
from loguru import logger

import seaborn

import icli.lang as lang
from icli.helpers import *  # FUT_EXP is appearing from here
from mutil.numeric import fmtPrice, fmtPricePad
from mutil.timer import Timer
import tradeapis.buylang as buylang

# Configure logger where the ib_insync live service logs get written.
# Note: if you have weird problems you don't think are being exposed
# in the CLI, check this log file for what ib_insync is actually doing.
logging.basicConfig(
    level=logging.INFO,
    filename=f"icli-{pendulum.now('US/Eastern')}.log",
    format="%(asctime)s %(message)s",
)

pp = pprint.PrettyPrinter(indent=4)

# setup color gradients we use to show gain/loss of daily quotes
COLOR_COUNT = 100
# palette 'RdYlGn' is a spectrum from low RED to high GREEN which matches
# the colors we want for low/negative (red) to high/positive (green)
MONEY_COLORS = seaborn.color_palette("RdYlGn", n_colors=COLOR_COUNT, desat=1).as_hex()

# only keep lowest 25 and highest 25 elements since middle values are less distinct
MONEY_COLORS = MONEY_COLORS[:25] + MONEY_COLORS[-25:]

# display order we want: RTY, ES, NQ, YM
FUT_ORD = dict(MES=-9, ES=-9, RTY=-10, M2K=-10, NQ=-8, MNQ=-8, MYM=-7, YM=-7)

# A-Z, Z-A, translate between them (lowercase only)
ATOZ = "".join([chr(x) for x in range(ord("a"), ord("z") + 1)])
ZTOA = ATOZ[::-1]
ATOZTOA_TABLE = str.maketrans(ATOZ, ZTOA)


def invertstr(x):
    return x.translate(ATOZTOA_TABLE)


# Fields updated live for toolbar printing.
# Printed in the order of this list (the order the dict is created)
# Some math and definitions for values:
# https://www.interactivebrokers.com/en/software/tws/usersguidebook/realtimeactivitymonitoring/available_for_trading.htm
# https://ibkr.info/node/1445
LIVE_ACCOUNT_STATUS = [
    # row 1
    "AvailableFunds",
    "BuyingPower",
    "Cushion",
    "DailyPnL",
    "DayTradesRemaining",
    # The API returns these, but ib_insync isn't allowing them yet.
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

STATUS_FIELDS = set(LIVE_ACCOUNT_STATUS)


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


def readableHTML(html):
    """Return contents of 'html' with tags stripped and in a _reasonably_
    readable plain text format"""

    return re.sub(r"(\n[\s]*)+", "\n", bs4.BeautifulSoup(html).get_text())


# logger.remove()
# logger.add(asink, colorize=True)

# Create prompt object.
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory, ThreadedHistory
from prompt_toolkit.application import get_app
from prompt_toolkit.shortcuts import set_title
import asyncio
import os

stocks = ["IWM", "QQQ", "VXX", "AAPL", "SBUX", "TSM"]

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
    "GLOBEX": ["ES", "RTY", "MNQ", "GBP"],  # "HE"],
    "ECBOT": ["YM"],  # , "TN", "ZF"],
    #    "NYMEX": ["GC", "QM"],
}

# Note: ContFuture is only for historical data; it can't quote or trade.
# So, all trades must use a manual contract month (quarterly)
futures = [
    Future(symbol=sym, lastTradeDateOrContractMonth=FUT_EXP, exchange=x, currency="USD")
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

    # initialized to True/False when we first see the account
    # ID returned from the API which will tell us if this is a
    # sandbox ID or True Account ID
    isSandbox: Optional[bool] = None

    # The Connection
    ib: IB = field(default_factory=IB)

    # generic cache for data usage (strikes, etc)
    cache: Mapping[Any, Any] = field(
        default_factory=lambda: diskcache.Cache("./cache-multipurpose")
    )

    # State caches
    quoteState: dict[str, Ticker] = field(default_factory=dict)
    quoteContracts: dict[str, Contract] = field(default_factory=dict)
    depthState: dict[Contract, Ticker] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)
    position: dict[str, float] = field(default_factory=dict)
    order: dict[str, float] = field(default_factory=dict)
    liveBars: dict[str, RealTimeBarList] = field(default_factory=dict)
    pnlSingle: dict[str, PnLSingle] = field(default_factory=dict)
    exiting: bool = False
    ol: buylang.OLang = field(default_factory=buylang.OLang)

    # Specific dict of ONLY fields we show in the live account status toolbar.
    # Saves us from sorting/filtering self.summary() with every full bar update.
    accountStatus: dict[str, float] = field(
        default_factory=lambda: dict(
            zip(LIVE_ACCOUNT_STATUS, [None] * len(LIVE_ACCOUNT_STATUS))
        )
    )

    # Cache all contractIds to names
    conIdCache: Mapping[int, Contract] = field(
        default_factory=lambda: diskcache.Cache("./cache-contracts")
    )

    def __post_init__(self):
        # just use the entire IBKRCmdlineApp as our app state!
        self.opstate = self

    async def qualify(self, *contracts) -> Union[list[Contract], None]:
        """Qualify contracts against the IBKR allowed symbols.

        Mainly populates .localSymbol and .conId

        We also cache the results for ease of re-use and for mapping
        contractIds back to names later."""

        # Note: this is the ONLY place we use self.ib.qualifyContractsAsync().
        # All other usage should use self.qualify() so the cache is maintained.
        got = await self.ib.qualifyContractsAsync(*contracts)

        # iterate resolved contracts and save them all
        for contract in got:
            # Populate the id to contract cache!
            if contract.conId not in self.conIdCache:
                # default 30 day expiration...
                self.conIdCache.set(contract.conId, contract, expire=86400 * 30)

        return got

    def contractForPosition(
        self, sym, qty: Optional[float] = None
    ) -> Union[None, tuple[Contract, float, float]]:
        """Returns matching portfolio position as (contract, size, marketPrice).

        Looks up position by symbol name and returns either provided quantity or total quantity.
        If no input quantity, return total position size.
        If input quantity larger than position size, returned size is capped to max position size."""
        portitems = self.ib.portfolio()
        logger.info("Port is: {}", portitems)
        contract = None
        for pi in portitems:
            # Note: using 'localSymbol' because for options, it includes
            # the full OCC-like format, while contract.symbol will just
            # be the underlying equity symbol.
            if pi.contract.localSymbol == sym:
                contract = pi.contract

                if qty is None:
                    qty = pi.position
                elif abs(qty) > abs(pi.position):
                    qty = pi.position

                return contract, qty, pi.marketPrice

        return None

    async def contractForOrderRequest(
        self, oreq: buylang.OrderRequest, exchange="SMART"
    ) -> Optional[Contract]:
        """Return a valid qualified contract for any order request.

        If order request has multiple legs, returns a Bag contract representing the spread.
        If order request only has one symbol, returns a regular future/stock/option contract.

        If symbol(s) in order request are not valid, returns None."""

        if oreq.isSpread():
            return await self.bagForSpread(oreq, exchange)

        if oreq.isSingle():
            contract = contractForName(oreq.orders[0].symbol, exchange=exchange)
            await self.qualify(contract)

            # only return success if the contract validated
            if contract.conId:
                return contract

            return None

        # else, order request had no orders...
        return None

    async def bagForSpread(
        self, oreq: buylang.OrderRequest, exchange="SMART", currency="USD"
    ) -> Optional[Bag]:
        """Given a multi-leg OrderRequest, return a qualified Bag contract.

        If legs do not validate, returns None and prints errors along the way."""

        # For IBKR spreads ("Bag" contracts), each leg of the spread is qualified
        # then placed in the final contract instead of the normal approach of qualifying
        # the final contract itself (because Bag contracts have Legs and each Leg is only
        # a contractId we have to look up via qualify() individually).
        contracts = [
            contractForName(s.symbol, exchange=exchange, currency=currency)
            for s in oreq.orders
        ]
        await self.qualify(*contracts)

        if not all(c.conId for c in contracts):
            logger.error("Not all contracts qualified!")
            return None

        contractUnderlying = contracts[0].symbol
        reqUnderlying = oreq.orders[0].underlying()
        if contractUnderlying != reqUnderlying.lstrip("/"):
            logger.error(
                "Resolved symbol [{}] doesn't match order underlying [{}]?",
                contractUnderlying,
                reqUnderlying,
            )
            return None

        if not all(c.symbol == contractUnderlying for c in contracts):
            logger.error("All contracts must have same underlying for spread!")
            return None

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
            symbol=contractUnderlying,
            exchange=useExchange or exchange,
            comboLegs=legs,
            currency=currency,
        )

    def midpointBracketBuyOrder(
        self,
        isLong: bool,
        qty: int,
        ask: float,
        stopPct: float,
        profitPts: float = None,
        stopPts: float = None,
    ):
        """Place a 3-sided order:
        - Market with Protection to buy immediately (long)
        - Profit taker: TRAIL LIT with trailStopPrice = (current ask + profitPts)
        - Stop loss: STP PRT with trailStopPrice = (current ask - stopPts)
        """

        lower, upper = boundsByPercentDifference(ask, stopPct)
        if isLong:
            lossPrice = lower
            trailStop = makeQuarter(ask - lower)

            openLimit = ask + 1

            openAction = "BUY"
            closeAction = "SELL"
        else:
            lossPrice = upper
            trailStop = makeQuarter(upper - ask)

            openLimit = ask - 1

            openAction = "SELL"
            closeAction = "BUY"

        # TODO: up/down One-Cancels-All brackets:
        #         BUY if +5 pts, TRAIL STOP 3 PTS
        #         SELL if -5 pts, TRAIL STOP 3 PTS
        if True:
            # Note: these orders require MANUAL order ID because by default,
            #       the order ID is populated on .placeOrder(), but we need to
            #       reference it here for the seconday order to reference
            #       the parent order!
            parent = Order(
                orderId=self.ib.client.getReqId(),
                action=openAction,
                totalQuantity=qty,
                transmit=False,
                # orderType="MKT PRT",
                orderType="LMT",
                lmtPrice=openLimit,
                outsideRth=True,
                tif="GTC",
            )

            profit = Order(
                orderId=self.ib.client.getReqId(),
                action=closeAction,
                totalQuantity=qty,
                parentId=parent.orderId,
                transmit=True,
                orderType="TRAIL LIMIT",
                outsideRth=True,
                tif="GTC",
                trailStopPrice=lossPrice,  # initial trigger level if price falls immediately
                lmtPriceOffset=0.75,  # price offset for the limit order when stop triggers
                auxPrice=trailStop,  # trailing amount before stop triggers
            )

            loss = Order(
                action=closeAction,
                totalQuantity=qty,
                parentId=parent.orderId,
                transmit=True,
                orderType="STP PRT",
                auxPrice=lossPrice,
            )

            return [parent, profit]  # , loss]

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
                        np.sign(positionSize) * -1 * t.order.lmtPrice,
                    )
                )

        # if only one and it's the full position, return without formatting
        if len(ts) == 1:
            if abs(int(positionSize)) == abs(ts[0][0]):
                return ts[0][1]

        # else, break out by order size, sorted from smallest to largest exit prices
        return sorted(ts, key=lambda x: abs(x[1]))

    def currentQuote(self, sym) -> Optional[Tuple[float, float]]:
        q = self.quoteState[sym.upper()]
        ago = (self.now - (q.time or self.now)).as_interval()
        show = [
            f"{q.contract.symbol}: bid {q.bid:,.2f} x {q.bidSize}",
            f"ask {q.ask:,.2f} x {q.askSize}",
            f"mid {(q.bid + q.ask) / 2:,.2f}",
            f"last {q.last:,.2f} x {q.lastSize}",
            f"ago {str(ago)}",
        ]
        logger.info("    ".join(show))

        # if no quote yet (or no prices available), return nothing...
        if all(np.isnan([q.bid, q.ask])) or (q.bid <= 0 and q.ask <= 0):
            return None

        return q.bid, q.ask

    def updatePosition(self, pos):
        self.position[pos.contract.symbol] = pos

    def updateOrder(self, trade):
        self.order[trade.contract.symbol] = trade

        # Only print update if this is regular runtime and not
        # the "load all trades on startup" cycle
        if self.connected:
            logger.warning("Order update: {}", trade)

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
            logger.error(
                "API Error [orderId {}] [code {}]: {}{}",
                reqId,
                errorCode,
                errorString,
                f" for {contract}" if contract else "",
            )

    def cancelHandler(self, err):
        logger.warning("Order canceled: {}", err)

    def commissionHandler(self, trade, fill, report):
        # Only report commissions if connected (not when loading startup orders)
        if not self.connected:
            logger.warning("Ignoring commission because not connected...")
            return

        # TODO: different sounds if PNL is a loss?
        #       different sounds for big wins vs. big losses?
        if fill.execution.side == "BOT":
            pygame.mixer.music.play()
        elif fill.execution.side == "SLD":
            pygame.mixer.music.play()

        logger.warning(
            "Order {} commission: {} {} {} at {} (total {} of {}) (commission {} ({} each)){}",
            fill.execution.orderId,
            fill.execution.side,
            fill.execution.shares,
            fill.contract.localSymbol,
            locale.currency(fill.execution.price),
            fill.execution.cumQty,
            trade.order.totalQuantity,
            locale.currency(fill.commissionReport.commission),
            locale.currency(fill.commissionReport.commission / fill.execution.shares),
            f" (pnl {locale.currency(fill.commissionReport.realizedPNL)})"
            if fill.commissionReport.realizedPNL
            else "",
        )

    def newsBHandler(self, news: NewsBulletin):
        logger.warning("News Bulletin: {}", readableHTML(news.message))

    def newsTHandler(self, news: NewsTick):
        logger.warning("News Tick: {}", news)

    def orderExecuteHandler(self, trade, fill):
        logger.warning("Trade executed for {}", fill.contract.localSymbol)
        if fill.execution.cumQty > 0:
            if trade.contract.conId not in self.pnlSingle:
                self.pnlSingle[trade.contract.conId] = self.ib.reqPnLSingle(
                    self.accountId, "", trade.contract.conId
                )
        else:
            # if quantity is gone, stop listening for updates and remove.
            self.ib.cancelPnLSingle(self.pnlSingle[trade.contract.conId])
            del self.pnlSingle[trade.contract.conId]

    def tickersUpdate(self, tickr):
        logger.info("Ticker update: {}", tickr)

    def updateSummary(self, v):
        """Each row is populated after connection then continually
        updated via subscription while the connection remains active."""
        # logger.info("Updating sumary... {}", v)
        self.summary[v.tag] = v.value

        # regular accounts are U...; sanbox accounts are DU... (apparently)
        # Some fields are for "All" accounts under this login, which don't help us here.
        # TODO: find a place to set this once instead of checking every update?
        if self.isSandbox is None and v.account != "All":
            self.isSandbox = v.account.startswith("D")

        if v.tag in STATUS_FIELDS:
            try:
                self.accountStatus[v.tag] = float(v.value)
            except:
                # don't care, just keep going
                pass

    def updatePNL(self, v):
        """Kinda like summary, except account PNL values aren't summary events,
        they are independent PnL events. shrug.

        Also note: we merge these into our summary dict instead of maintaining
        an indepdent PnL structure."""

        # TODO: keep moving average of daily PNL and trigger sounds/events
        #       if it spikes higher/lower.
        # logger.info("Updating PNL... {}", v)
        self.summary["UnrealizedPnL"] = v.unrealizedPnL
        self.summary["RealizedPnL"] = v.realizedPnL
        self.summary["DailyPnL"] = v.dailyPnL

        try:
            self.accountStatus["UnrealizedPnL"] = float(v.unrealizedPnL)
            self.accountStatus["RealizedPnL"] = float(v.realizedPnL)
            self.accountStatus["DailyPnL"] = float(v.dailyPnL)
        except:
            # don't care, just keep going
            pass

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
        self.now = pendulum.now()

        def fmtPrice2(n: float):
            # Some prices may not be populated if they haven't
            # happened yet (e.g. PNL values if no trades for the day yet, etc)
            if not n:
                n = 0

            # if GTE $1 million, stop showing cents.
            if n > 999_999.99:
                return f"{n:>10,.0f}"

            return f"{n:>10,.2f}"

        def fmtPriceOpt(n):
            if n:
                # assume trading $0.01 to $99.99 range for options
                return f"{n:>5,.2f}"

            return f"{n:>5}"

        # Fields described at:
        # https://ib-insync.readthedocs.io/api.html#module-ib_insync.ticker
        def formatTicker(c):
            usePrice = c.marketPrice()
            ago = (self.now - (c.time or self.now)).as_interval()
            try:
                percentUpFromLow = (
                    abs(usePrice - c.low) / ((usePrice + c.low) / 2)
                ) * 100
                percentUpFromClose = (
                    ((usePrice - c.close) / ((usePrice + c.close) / 2)) * 100
                    if c.close
                    else 0
                )
            except:
                # price + (low or close) is zero... can't do that.
                percentUpFromLow = 0
                percentUpFromClose = 0

            def mkcolor(
                n: float, vals: Union[str, list[str]], colorRanges: list[str]
            ) -> Union[str, list[str]]:
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
            b_s = (
                f"{c.bidSize:>6,}"
                if (c.bidSize < 100_000 or np.isnan(c.bidSize))
                else f"{c.bidSize // 1000:>5}k"
            )
            a_s = (
                f"{c.askSize:>6,}"
                if (c.askSize < 100_000 or np.isnan(c.askSize))
                else f"{c.askSize // 1000:>5}k"
            )

            bigboi = (len(c.contract.localSymbol) > 15) or c.contract.comboLegs

            if bigboi:
                # if c.modelGreeks:
                #     mark = c.modelGreeks.optPrice

                if c.bid and c.bidSize and c.ask and c.askSize:
                    # weighted sum of bid/ask as midpoint
                    mark = ((c.bid * c.bidSize) + (c.ask * c.askSize)) / (
                        c.bidSize + c.askSize
                    )
                else:
                    # IBKR reports "no bid" as -1. le sigh.
                    mark = (c.bid + c.ask) / 2 if c.bid > 0 else c.ask / 2

                # For options, instead of using percent difference between
                # prices, we use percent return over the low/close instead.
                # e.g. if low is 0.05 and current is 0.50, we want to report
                #      a 900% multiple, not a 163% difference between the
                #      two numbers as we would report for normal stock price changes.
                # Also note: we use 'mark' here because after hours, IBKR reports
                # the previous day closing price as the current price, which clearly
                # isn't correct since it ignores the entire most recent day.
                bighigh = ((mark / c.high if c.high else 1) - 1) * 100
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
                else:
                    und = None
                    underlyingStrikeDifference = None
                    iv = None

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
                    for x in c.contract.comboLegs:
                        contract = self.conIdCache[x.conId]
                        rns.append(
                            f"{x.action[0]} {x.ratio:2} {contract.localSymbol or contract.symbol}"
                        )

                    rowName = "\n".join(rns)
                    return " ".join(
                        [
                            rowName,
                            f"{fmtPriceOpt(mark):>6}",
                            f" {fmtPriceOpt(c.bid):>} x {b_s}   {fmtPriceOpt(c.ask):>} x {a_s} ",
                            "HALTED!" if c.halted > 0 else "",
                        ]
                    )
                else:
                    rowName = f"{c.contract.localSymbol or c.contract.symbol:<6}:"

                    return " ".join(
                        [
                            rowName,
                            f"[u {fmtPricePad(und, 8)} ({underlyingStrikeDifference or -0:>6,.2f}%)]",
                            f"[iv {iv or 0:.2f}]",
                            f"{fmtPriceOpt(mark)}",
                            # f"{fmtPriceOpt(usePrice)}",
                            f"({pctBigHigh} {amtBigHigh} {fmtPriceOpt(c.high)})",
                            f"({pctBigLow} {amtBigLow} {fmtPriceOpt(c.low)})",
                            f"({pctBigClose} {amtBigClose} {fmtPriceOpt(c.close)})",
                            #                        f"[h {fmtPriceOpt(c.high)}]",
                            #                        f"[l {fmtPriceOpt(c.low)}]",
                            f" {fmtPriceOpt(c.bid)} x {b_s}   {fmtPriceOpt(c.ask)} x {a_s} ",
                            #                        f"[c {fmtPriceOpt(c.close)}]",
                            f"  ({str(ago)})",
                            "HALTED!" if c.halted > 0 else "",
                        ]
                    )

            pctUpLow, amtUpLow = mkPctColor(
                percentUpFromLow,
                [
                    f"{percentUpFromLow:>5.2f}%",
                    f"{amtLow:>6.2f}" if amtLow < 1000 else f"{amtLow:>6.0f}",
                ],
            )
            pctUpClose, amtUpClose = mkPctColor(
                percentUpFromClose,
                [
                    f"{percentUpFromClose:>6.2f}%",
                    f"{amtClose:>7.2f}" if amtLow < 1000 else f"{amtClose:>7.0f}",
                ],
            )

            return f"{c.contract.localSymbol or c.contract.symbol:<7}: {fmtPricePad(usePrice)}  ({pctUpLow} {amtUpLow}) ({pctUpClose} {amtUpClose}) {fmtPricePad(c.high)}   {fmtPricePad(c.low)} {fmtPricePad(c.bid)} x {b_s} {fmtPricePad(c.ask)} x {a_s}  {fmtPricePad(c.open)} {fmtPricePad(c.close)}    ({str(ago)})"

        try:
            pass
            # logger.info("One future: {}", self.quoteState["ES"].dict())
        except:
            pass

        try:
            rowlen, _ = shutil.get_terminal_size()

            rowvals = [[]]
            currentrowlen = 0
            DT = []
            for cat, val in self.accountStatus.items():
                # if val == 0:
                #    continue

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
                        csv = ",".join([str(x) for x in DT])
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
                if c.secType == "FUT":
                    priority = FUT_ORD[c.symbol] if c.symbol in FUT_ORD else 0
                    return (0, priority, c.symbol)

                if c.secType == "OPT":
                    # options are medium last because they are wide
                    priority = 0
                    return (2, priority, c.localSymbol)

                if c.secType == "FOP":
                    # future options are above other options...
                    priority = -1
                    return (2, priority, c.localSymbol)

                if c.secType == "BAG":
                    # bags are last because their descriptions are big
                    priority = 0
                    return (3, priority, c.symbol)

                # else, just by name.
                # BUT we do these in REVERSE order since they
                # are at the end of the table!
                # (We create "reverse order" by translating all
                #  letters into their "inverse" where a == z, b == y, etc).
                priority = 0
                return (1, priority, invertstr(c.symbol.lower()))

            now = str(pendulum.now("US/Eastern"))

            return HTML(
                f"""{now}\n"""
                + "\n".join(
                    [
                        formatTicker(quote)
                        for sym, quote in sorted(
                            self.quoteState.items(), key=sortQuotes
                        )
                    ]
                )
                + "\n"
                + balrows
            )
        except:
            logger.exception("qua?")
            return HTML("No data yet...")  # f"""{self.now:<40}\n""")

    async def qask(self, terms) -> Union[dict[str, Any], None]:
        """Ask a questionary survey using integrated existing toolbar showing"""
        result = dict()
        extraArgs = dict(bottom_toolbar=self.bottomToolbar, refresh_interval=0.750)
        for t in terms:
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

    async def dorepl(self):
        # Setup...

        # wait until we start getting data from the gateway...
        loop = asyncio.get_event_loop()

        dispatch = lang.Dispatch()
        pygame.mixer.init()

        # TODO: could probably just be: pathlib.Path(__file__).parent
        pygame.mixer.music.load(
            pathlib.Path(os.path.abspath(__file__)).parent / "CANYON.MID"
        )

        contracts = [Stock(sym, "SMART", "USD") for sym in stocks]
        contracts += futures

        def requestMarketData():
            logger.info("Requesting market data...")
            for contract in contracts:
                # Additional details can be requested:
                # https://ib-insync.readthedocs.io/api.html#ib_insync.ib.IB.reqMktData
                # https://interactivebrokers.github.io/tws-api/tick_types.html
                # By default, only common fields are populated (so things like 13/26/52 week
                # highs and lows aren't created unless requested via tick set 165, etc)
                # Also can subscribe to live news feed per symbol with tick 292 (news result
                # returned via tickNewsEvent callback, we think)

                # Tell IBKR API to return "last known good quote" if outside
                # of regular market hours instead of giving us bad data.
                # Must be called before each market data request!
                self.ib.reqMarketDataType(2)

                tickFields = tickFieldsForContract(contract)
                self.quoteState[contract.symbol] = self.ib.reqMktData(
                    contract, tickFields
                )

                self.quoteContracts[contract.symbol] = contract

                # Note: the Ticker returned by reqMktData() is updated in-place, so we can just
                # read the object on a timer for the latest value(s)

        async def reconnect():
            # don't reconnect if an exit is requested
            if self.exiting:
                return

            logger.info("Connecting to IBKR API...")
            while True:
                try:
                    self.connected = False

                    # NOTE: Client ID *MUST* be 0 to allow modification of
                    #       existing orders (which get "re-bound" with a new
                    #       order id when client 0 connects—but it *only* works
                    #       for client 0)

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
                    # self.ib.pendingTickersEvent += self.tickersUpdate

                    # openOrderEvent is noisy and randomly just re-submits
                    # already static order details as new events.
                    # self.ib.openOrderEvent += self.orderOpenHandler
                    self.ib.execDetailsEvent += self.orderExecuteHandler

                    await self.ib.connectAsync(
                        self.host,
                        self.port,
                        clientId=0,
                        readonly=False,
                        account=self.accountId,
                    )

                    logger.info(
                        "Connected! Current Request ID: {}", self.ib.client._reqIdSeq
                    )

                    self.connected = True

                    self.ib.reqNewsBulletins(True)

                    requestMarketData()

                    # reset cached states on reconnect so we don't have stale
                    # data by mistake
                    self.summary.clear()
                    self.position.clear()
                    self.order.clear()
                    self.pnlSingle.clear()

                    # Note: "PortfolioEvent" is fine here since we are using a single account.
                    # If you have multiple accounts, you want positionEvent (the IBKR API
                    # doesn't allow "Portfolio" to span accounts, but Positions can be reported
                    # from multiple accounts with one API connection apparently)
                    self.ib.updatePortfolioEvent += lambda row: self.updatePosition(row)

                    # request live updates (well, once per second) of account and position values
                    self.ib.reqPnL(self.accountId)

                    # Subscribe to realtime PnL updates for all positions in account
                    # Note: these are updated once per second per position! nice.
                    # TODO: add this to the account order/filling notifications too.
                    for p in self.ib.portfolio():
                        self.pnlSingle[p.contract.conId] = self.ib.reqPnLSingle(
                            self.accountId, "", p.contract.conId
                        )

                    lookupBars = [
                        Future(
                            symbol="MES",
                            exchange="GLOBEX",
                            lastTradeDateOrContractMonth=FUT_EXP,
                        ),
                        Future(
                            symbol="MNQ",
                            exchange="GLOBEX",
                            lastTradeDateOrContractMonth=FUT_EXP,
                        ),
                    ]

                    if False:
                        self.liveBars = {
                            c.symbol: self.ib.reqRealTimeBars(c, 5, "TRADES", False)
                            for c in lookupBars
                        }

                    # run some startup accounting subscriptions concurrently
                    await asyncio.gather(
                        self.ib.reqAccountSummaryAsync(),  # self.ib.reqPnLAsync()
                    )
                    break
                except:
                    logger.error("Failed to connect to IB Gateway, trying again...")
                    # logger.exception("why?")
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

        set_title(f"{self.levelName().title()} Trader")
        self.ib.disconnectedEvent += lambda: asyncio.create_task(reconnect())

        session = PromptSession(
            history=ThreadedHistory(
                FileHistory(
                    os.path.expanduser(f"~/.tplatcli_ibkr_history.{self.levelName()}")
                )
            )
        )

        app = session.app

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

        # Primary REPL loop
        while True:
            try:
                text1 = await session.prompt_async(
                    f"{self.levelName()}> ",
                    bottom_toolbar=self.bottomToolbar,
                    # refresh_interval=3,
                    # mouse_support=True,
                    # completer=completer, # <-- causes not to be full screen due to additional dropdown space
                    complete_in_thread=True,
                    complete_while_typing=True,
                )

                # Attempt to run the command submitted into the prompt
                cmd, *rest = text1.split(" ", 1)
                with Timer(cmd):
                    result = await dispatch.runop(
                        cmd, rest[0] if rest else None, self.opstate
                    )

                continue

                # Below are legacy commands either not copied over to lang.py
                # yet, or were copied and we forgot to delete here.
                # This was the original (temporary) command implementation
                # before we used the mutil/dispatch.py abstraction.
                if text1.startswith("fast"):
                    try:
                        cmd, symbol, action, qty, price = text1.split()
                    except:
                        logger.warning("Format: symbol BUY|SELL qty price")
                        continue

                    action = action.upper()

                    if qty.startswith("$"):
                        dollarSpend = int(qty[1:])
                        assetPrice = float(price)

                        # if is option, adjust for contract size...
                        if len(symbol) > 15:
                            assetPrice *= 100

                        qty = dollarSpend // assetPrice

                    if action not in {"BUY", "SELL"}:
                        logger.error(
                            "Action must be BUY or SELL: symbol, action, qty, price"
                        )
                        continue

                    contract = contractForName(symbol)
                    logger.info(
                        "Placing order for {} {} at {} DAY via {}",
                        qty,
                        symbol,
                        price,
                        contract,
                    )

                    order = self.limitOrder(action, int(qty), float(price))
                    trade = self.ib.placeOrder(contract, order)
                    logger.info("Placed: {}", pp.pformat(trade))
                elif text1 == "lmt":
                    ...
                elif text1.startswith("rcheck "):
                    cmd, base, pct = text1.split()
                    base = float(base)
                    pct = float(pct) / 100

                    lower, upper = boundsByPercentDifference(base, pct)
                    lowmid = base - lower
                    logger.info(
                        "Range of {:,.2f} ± {:.4f}% is ({:,.2f}, {:,.2f}) with distance ±{:,.2f}",
                        base,
                        pct * 100,
                        lower,
                        upper,
                        lowmid,
                    )
                elif text1.startswith("fu"):
                    cmd, *rest = text1.split()

                    if rest:
                        try:
                            symbol, side, qty, trail, *check = rest
                        except:
                            logger.error(
                                "fu [symbol] [side] [qty] [trail] [[checkonly]]"
                            )
                            continue

                        got = dict(Symbol=symbol, Side=side, Quantity=qty)
                        got["Percentage Stop / Trail"] = trail
                        got["Place Order"] = "Check Only" if check else "Live Order"
                    else:
                        stuff = [
                            Q("Symbol"),
                            Q("Side", choices=["Buy", "Sell"]),
                            # Q("Order", choices=["LMT", "MKT", "STP"]),
                            Q("Quantity"),
                            Q("Percentage Stop / Trail"),
                            Q("Place Order", choices=["Check Only", "Live Order"]),
                        ]
                        got = await self.qask(stuff)
                        logger.info("Got: {}", got)

                    try:
                        sym = got["Symbol"].upper()
                        qty = got["Quantity"]
                        isLong = got["Side"].title() == "Buy"
                        liveOrder = got["Place Order"] == "Live Order"
                        percentStop = float(got["Percentage Stop / Trail"]) / 100
                        fxchg = FUTS_EXCHANGE[sym]
                    except:
                        logger.info("Canceled by lack of fields...")
                        continue

                    bid, ask = self.currentQuote(sym)
                    (qualified,) = await self.qualify(
                        Future(
                            currency="USD",
                            symbol=sym,
                            exchange=fxchg.exchange,
                            lastTradeDateOrContractMonth=FUT_EXP,
                        )
                    )
                    logger.info("qual: {} for {}", qualified, fxchg.name)

                    order = self.midpointBracketBuyOrder(
                        qty=qty, isLong=isLong, ask=ask, stopPct=percentStop
                    )

                    if liveOrder:
                        placed = []
                        for o in order:
                            logger.info("Placing order: {}", o)
                            t = self.ib.placeOrder(qualified, o)
                            placed.append(t)

                        logger.info("Placed: {}", placed)
                    else:
                        for o in order:
                            logger.info("Checking order: {}", o)
                            # what-if orders always transmit true, but they aren't live.
                            o.transmit = True
                            ordstate = await self.ib.whatIfOrderAsync(qualified, o)
                            logger.info("ordstate: {}", ordstate)

                elif text1 == "bars":
                    logger.info("bars: {}", self.liveBars)

                    # reset bar cache so it doesn't grow forever...
                    for k, v in self.liveBars.items():
                        v.clear()
                elif text1.lower().startswith("buy "):
                    cmd, *rest = text1.split()
                    symbol, qty, algo, profit, stop = rest
                    qty = int(qty)
                    profit = float(profit)
                    stop = float(stop)
                    contract = Future(
                        currency="USD",
                        symbol="MES",
                        lastTradeDateOrContractMonth=FUT_EXP,
                        exchange="GLOBEX",
                    )
                    order = self.midpointBracketBuyOrder(1, 5, 0.75)
                elif text1 == "try":
                    logger.info("Ordering...")
                    logger.info("QS: {}", self.quoteState["ES"])
                    contract = Stock("AMD", "SMART", "USD")
                    order = LimitOrder(
                        action="BUY",
                        totalQuantity=500,
                        lmtPrice=33.33,
                        algoStrategy="Adaptive",
                        algoParams=[TagValue("adaptivePriority", "Urgent")],
                    )
                    ordstate = await self.ib.whatIfOrderAsync(contract, order)
                    logger.info("ordstate: {}", ordstate)
                elif text1 == "tryf":
                    logger.info("Ordering...")
                    contract = Future(
                        currency="USD",
                        symbol="MES",
                        lastTradeDateOrContractMonth=FUT_EXP,
                        exchange="GLOBEX",
                    )
                    order = LimitOrder(
                        action="BUY",
                        totalQuantity=770,
                        lmtPrice=400.75,
                        algoStrategy="Adaptive",
                        algoParams=[TagValue("adaptivePriority", "Urgent")],
                    )
                    ordstate = await self.ib.whatIfOrderAsync(contract, order)
                    logger.info("ordstate: {}", ordstate)

                if not text1:
                    continue
            except KeyboardInterrupt:
                continue  # Control-C pressed. Try again.
            except EOFError:
                logger.error("Exiting...")
                self.exiting = True
                break  # Control-D pressed.
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
                        asyncio.sleep(1)
                        pass

    def stop(self):
        self.ib.disconnect()

    async def setup(self):
        pass
