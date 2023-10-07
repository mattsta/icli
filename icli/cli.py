#!/usr/bin/env python3

original_print = print
import datetime
import decimal

import fnmatch  # for glob string matching!

# for automatic money formatting in some places
import locale
import math
import os

import pathlib
import re
import sys

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import bs4

# http://www.grantjenks.com/docs/diskcache/
import diskcache

import numpy as np

import pandas as pd

import pendulum
from prompt_toolkit import Application, print_formatted_text
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

# from prompt_toolkit import print_formatted_text as print
from prompt_toolkit.formatted_text import HTML

import icli.orders as orders

from icli.futsexchanges import FUTS_EXCHANGE

from . import agent

locale.setlocale(locale.LC_ALL, "")

import asyncio

import logging
import os

import ib_insync

# sounds!

# Tell pygame to not print a hello message when it is imported
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame

import seaborn
from ib_insync import (
    Bag,
    ComboLeg,
    Contract,
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
from icli.helpers import *  # FUT_EXP is appearing from here
import tradeapis.buylang as buylang
from mutil.numeric import fmtPrice, fmtPricePad
from mutil.timer import Timer


# Configure logger where the ib_insync live service logs get written.
# Note: if you have weird problems you don't think are being exposed
# in the CLI, check this log file for what ib_insync is actually doing.
logging.basicConfig(
    level=logging.INFO,
    filename=f"icli-{pendulum.now('US/Eastern')}.log",
    format="%(asctime)s %(message)s",
)

import prettyprinter as pp

pp.install_extras(["dataclasses"], warn_on_error=False)

# setup color gradients we use to show gain/loss of daily quotes
COLOR_COUNT = 100
# palette 'RdYlGn' is a spectrum from low RED to high GREEN which matches
# the colors we want for low/negative (red) to high/positive (green)
MONEY_COLORS = seaborn.color_palette("RdYlGn", n_colors=COLOR_COUNT, desat=1).as_hex()

# only keep lowest 25 and highest 25 elements since middle values are less distinct
MONEY_COLORS = MONEY_COLORS[:25] + MONEY_COLORS[-25:]

# display order we want: RTY, ES / SPX, NQ / COMP, YM, Index ETFs
FUT_ORD = dict(
    MES=-9,
    ES=-9,
    SPY=-6,
    SPX=-9,
    NANOS=-9,
    RTY=-10,
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

    return re.sub(
        r"(\n[\s]*)+", "\n", bs4.BeautifulSoup(html, features="html.parser").get_text()
    )


import asyncio
import os

# logger.remove()
# logger.add(asink, colorize=True)

# Create prompt object.
from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.history import FileHistory, ThreadedHistory
from prompt_toolkit.shortcuts import set_title

stocks = ["IWM", "QQQ", "SPY", "UVXY", "AAPL", "SBUX", "TSM"]

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
    Index("VIF", "CBOE"),  # VIX Back-Month Component (far term)
    Index("VIX", "CBOE"),  # VIX Currently
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

    # count total toolbar refreshes
    updates: int = 0

    # True if use sound for trades...
    alert: bool = False

    # Events!
    scheduler: dict[str, Any] = field(default_factory=dict)

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

    # Cache all contractIds to names
    conIdCache: Mapping[int, Contract] = field(
        default_factory=lambda: diskcache.Cache("./cache-contracts")
    )

    def __post_init__(self) -> None:
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
                self.conIdCache.set(contract.conId, contract, expire=86400 * 30)  # type: ignore

        return got

    def contractsForPosition(
        self, sym, qty: Optional[float] = None
    ) -> list[tuple[Contract, float, float]]:
        """Returns matching portfolio positions as list of (contract, size, marketPrice).

        Note: input 'sym' can be a glob pattern for symbol matching. '?' matches single character, '*' matches any characters.

        Looks up position by symbol name (allowing globs) and returns either provided quantity or total quantity.
        If no input quantity, return total position size.
        If input quantity larger than position size, returned size is capped to max position size."""
        portitems = self.ib.portfolio()
        # logger.debug("Current Portfolio is: {}", portitems)

        results = []
        for pi in portitems:
            # Note: using 'localSymbol' because for options, it includes
            # the full OCC-like format, while contract.symbol will just
            # be the underlying equity symbol.
            # Note note: using fnmatch.filter() because we allow 'sym' to
            #            have glob characters for multiple lookups at once!
            # Note 3: options .localSymbols have the space padding, so remove for input compare.
            # TODO: fix temporary hack of OUR symbols being like /NQ but position values dont' have the slash...
            if fnmatch.filter(
                [pi.contract.localSymbol.replace(" ", "")], sym.replace("/", "")
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
            cgot = await self.qualify(contract)

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

        # Temporarily removed because it breaks with weekly index options
        if False:
            # FIX for SPX/SPXW
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

    def symbolNormalizeIndexWeeklyOptions(self, name: str) -> str:
        """Weekly index options have symbol names with 'W' but orders are placed without."""
        return name.replace("SPXW", "SPX").replace("RUTW", "RUT").replace("NDXP", "NDX")

    def quoteResolve(self, lookup: str) -> str:
        """Resolve a local symbol alias like ':33' to current symbol name for the index."""

        # TODO: this doesn't work for futures symbols. Probably need to read the contract type
        #       to re-apply or internal formatting? futs: /; CFD: CFD; crypto: C; ...
        return self.quotesPositional[int(lookup[1:])][1].contract.localSymbol.replace(
            " ", ""
        )

    async def placeOrderForContract(
        self,
        sym: str,
        isLong: bool,
        contract: Contract,
        qty: float,
        price: float,
        orderType: str,
        preview=False,
    ):
        """Place a BUY (isLong) or SELL (!isLong) for qualified 'contract' at qty/price.

        If 'qty' is negative we calculate a live quantity+price based
        on the (positive) number as a dollar value."""

        # Immediately ask to add quote to live quotes for this trade positioning...
        # turn option contract lookup into non-spaced version
        sym = sym.replace(" ", "")

        if price > 0:
            price = comply(contract, price)
            logger.info(
                "[{}] Request to order qty {:,.2f} price {:,.2f}", sym, qty, price
            )
        else:
            logger.info(
                "[{}] Request to order at dynamic qty/price: {:,.2f} price {:,.2f}",
                sym,
                qty,
                price,
            )

        # need to replace underlying if is "fake settled underlying"
        quotesym = sym  # self.symbolNormalizeIndexWeeklyOptions(sym)
        await self.dispatch.runop("add", f'"{quotesym}"', self.opstate)

        if not contract.conId:
            # spead contracts don't have IDs, so only reject if NOT a spread here.
            if contract.tradingClass != "COMB":
                logger.error(
                    "[{} :: {}] Not submitting order because contract not qualified!",
                    sym,
                    quotesym,
                )
                return None

        if isinstance(contract, (Option, Bag)) or contract.tradingClass == "COMB":
            # Purpose: don't trigger warning about "RTH option has no effect" with options...
            # TODO: check if RTH includes extended late 4:15 ending options SPY / SPX / QQQ / IWM / etc?
            if contract.localSymbol[0:3] in {"SPX", "VIX"}:
                # SPX and VIX options now trade 23/5 but anything not 0930-1600 (maybe even to 1615?) is
                # considered "outside RTH"
                outsideRth = True
            else:
                outsideRth = False
        else:
            # Algos can only operate RTH:
            if " " in orderType or (
                orderType
                in {"MIDPRICE", "MKT + ADAPTIVE + FAST", "LMT + ADAPTIVE + FAST"}
            ):
                outsideRth = False
            else:
                # REL and LMT/MKT/MOO/MOC orders can be outside RTH
                outsideRth = True

        # TODO: cleanup, also verify how we want to run FAST or EVICT outside RTH?
        if " " in orderType or (
            orderType in {"MIDPRICE", "MKT + ADAPTIVE + FAST", "LMT + ADAPTIVE + FAST"}
        ):
            outsideRth = False

        if isinstance(contract, Crypto) and isLong:
            # Crypto can only use IOC or Minutes for tif BUY
            # (but for SELL, can use IOC, Minutes, Day, GTC)
            tif = "Minutes"
        else:
            tif = "GTC"

        # Negative 'qty' is a dollar amount to buy instead of share/contract
        # quantity, so we fetch a live quote to determine the initial quantity.
        if qty < 0:
            # we treat negative quantities as dollar amounts (because
            # we convert 'qty' to float, so we can't pass through $3000, so
            # instead we do -3000 for $3,000).

            # also note: negative quantites are not shorts, shorts are specified
            # by SELL QTY, not SELL -QTY, not BUY -QTY.

            quoteKey = lang.lookupKey(contract)

            # if this is a new quote just requested, we may need to wait
            # for the system to populate it...
            loopFor = 10
            while not (currentQuote := self.currentQuote(quoteKey)):
                logger.warning(
                    "[{} :: {}] Waiting for quote to populate...", quoteKey, loopFor
                )
                try:
                    await asyncio.sleep(0.133)
                except:
                    logger.warning("Cancelled waiting for quote...")
                    return

                if (loopFor := loopFor - 1) == 0:
                    # if we exhausted the loop, we didn't get a usable quote so we can't
                    # do the requested price-based position sizing.
                    logger.error("Never received valid quote prices: {}", currentQuote)
                    return

            bid, ask, multiplier = currentQuote

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
            if isinstance(contract, Option):
                # Options retain "regular" midpoint behavior because spreads can be wide.
                mid = round(((bid + ask) / 2), 2)

                # if no bid (nan), just play off the ask.
                if mid != mid:
                    mid = round(ask / 2, 2)
            else:
                # equity, futures, etc get the wider margins
                mid = round(((bid + ask) / 2) * (1.01 if isLong else 0.99), 2)

            price = comply(contract, mid)

            # since this is in the "negative quantity" block, we convert the
            # negative number to a positive number for representing total
            # amount to spend.
            amt = abs(qty)

            # calculate order quantity for spend budget at current estimated price
            logger.info(
                "[{}] Trying to order ${:,.2f} worth at ${:,.2f}...", sym, amt, price
            )

            qty = self.quantityForAmount(contract, amt, price)

            if not qty:
                logger.error(
                    "[{}] Zero quantity calculated for: {} {} {}!",
                    sym,
                    contract,
                    amt,
                    price,
                )
                return None

            assert qty > 0

            # If integer, show integer, else show fractions.
            logger.info(
                "Ordering {:,} {} at ${:,.2f} for ${:,.2f}",
                qty,
                sym,
                price,
                qty * price * multiplier,
            )

        assert qty > 0

        order = orders.IOrder(
            "BUY" if isLong else "SELL", qty, price, outsiderth=outsideRth, tif=tif  # type: ignore
        ).order(orderType)

        if preview:
            logger.info(
                "[{}] PREVIEW REQUEST {} via {}",
                contract.localSymbol,
                contract,
                pp.pformat(order),
            )
            trade = await self.ib.whatIfOrderAsync(contract, order)
            logger.info(
                "[{}] PREVIEW RESULT: {}", contract.localSymbol, pp.pformat(trade)
            )

            # (if trade isn't valid, trade is an empty list, so only print valid objects...)
            if trade:
                # sigh, these are strings of course.
                excess = float(trade.equityWithLoanAfter) - float(trade.initMarginAfter)
                if excess < 0:
                    logger.warning(
                        "[{}] TRADE NOT VIABLE; MISSING EQUITY: ${:,.2f}",
                        contract.localSymbol,
                        excess,
                    )

            return False

        logger.info("[{}] Ordering {} via {}", contract.localSymbol, contract, order)
        trade = self.ib.placeOrder(contract, order)

        # TODO: add optional agent-like feature HERE to modify order in steps for buys (+price, -qty)
        #       or for sells (-price).
        # TODO: move order logic from "buy" lang.py cmd to its own agent feature.
        #       Needs: agent state logged to persistnet data structure, check events on callback for next event in graph (custom OTOCO, etc).
        logger.info(
            "[{} :: {} :: {}] Placed: {}",
            trade.orderStatus.orderId,
            trade.orderStatus.status,
            contract.localSymbol,
            pp.pformat(trade),
        )

        return order, trade

    def amountForTrade(
        self, trade: Trade
    ) -> tuple[float, float, float, Union[float, int]]:
        """Return dollar amount of trade given current limit price and quantity.

        Also compensates for contract multipliers correctly.

        Returns:
            - calculated remaining amount
            - calculated total amount
            - current limit price
            - current quantity remaining
        """

        currentPrice = trade.order.lmtPrice
        currentQty = trade.orderStatus.remaining
        totalQty = currentQty + trade.orderStatus.filled
        avgFillPrice = trade.orderStatus.avgFillPrice

        # If contract has multiplier (like 100 underlying per option),
        # calculate total spend with mul * p * q.
        # The default "no multiplier" value is '', so this check should be fine.
        if isinstance(trade.contract, Future):
            # FUTURES HACK BECAUSE WE DO EXTERNAL MARGIN CALCULATIONS REGARDLESS OF MULTIPLIER
            mul = 1
        else:
            mul = int(trade.contract.multiplier) if trade.contract.multiplier else 1

        # use average price IF fills have happened, else use current limit price
        return (
            currentQty * currentPrice * mul,
            totalQty * (avgFillPrice or currentPrice) * mul,
            currentPrice,
            currentQty,
        )

    def quantityForAmount(
        self, contract: Contract, amount: float, limitPrice: float
    ) -> Union[int, float]:
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
            mul = int(contract.multiplier) if contract.multiplier else 1
        else:
            mul = 1

        assert mul > 0

        # total spend amount divided by price of thing to buy == how many things to buy
        qty = amount / (limitPrice * mul)
        assert qty > 0

        if not isinstance(contract, Crypto):
            # only crypto orders support fractional quantities over the API.
            # TODO: if IBKR ever enables fractional shares over the API,
            #       we can make the above Crypto check for (Crypto, Stock).
            qty = math.floor(qty)

        return qty

    def midpointBracketBuyOrder(
        self,
        contract: Contract,
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
            trailStop = comply(contract, ask - lower)

            openLimit = ask + 1

            openAction = "BUY"
            closeAction = "SELL"
        else:
            lossPrice = upper
            trailStop = comply(contract, upper - ask)

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

    def currentQuote(self, sym) -> Optional[tuple[float, float]]:
        q = self.quoteState[sym.upper()]
        ago = (self.now - (q.time or self.now)).as_interval()

        assert q.contract
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

        # 'contract.multiplier' is a string and can also be an empty string sometimes,
        # so if it is empty, default to 1
        return (q.bid, q.ask, int(q.contract.multiplier or 1))

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
        #       different sounds for commission credit vs. large commission fee?

        # TODO: disable audio for algo trades?
        # TODO: figure out the bug ehre, sometimes if they play back-to-back too fast, the
        #       entire program locks up in a 100% CPU loop until manually kill -9'd?

        if self.alert:
            pygame.mixer.music.stop()

            if fill.execution.side == "BOT":
                pygame.mixer.music.play()
            elif fill.execution.side == "SLD":
                pygame.mixer.music.play()

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

        def fmtPriceOpt(n):
            if isinstance(n, (int, float)):
                # assume trading $0.01 to $99.99 range for options
                # (we can get intgers here is we have decided there's no valid bid
                #  and we're just marking a price to 0)
                return f"{n:>5,.2f}"

            return f"{n:>5}"

        def updateEMA(sym, price):
            # if no price, don't update.
            if (price <= 0) or (price != price):
                return

            for back in (100, 300):
                prev = self.ema[sym][back]

                # use previous price or initialize with current price
                if (prev <= 0) or (prev != prev):
                    prev = price

                # fmt: off
                # if prev != prev:
                #    logger.info("NaN in EMA? [{} :: {}] {} {} -> {}", prev, price, sym, back, self.ema[sym][back])
                # fmt: on

                k = 2 / (back + 1)
                self.ema[sym][back] = round(k * (price - prev) + prev, 2)

        def getEMA(sym, back):
            # If we don't have enough entries for the full EMA period yet,
            # just return the oldest entry in the recent observation capped collections.
            return self.ema[sym][back]

        # Fields described at:
        # https://ib-insync.readthedocs.io/api.html#module-ib_insync.ticker
        def formatTicker(c):
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
            ls = c.contract.localSymbol or c.contract.symbol

            if c.bid > 0 and c.bid == c.bid and c.ask > 0 and c.ask == c.ask:
                # TODO: add proper rounding for futures tick sizes
                usePrice = round((c.bid + c.ask) / 2, 2)
            else:
                usePrice = c.last if c.last == c.last else c.close

            if (c.high == c.high and c.low == c.low) or (c.bid > 0 and c.ask > 0):
                # only update EMA if this has price-like details and isn't TRIN/TICK/AD
                updateEMA(ls, usePrice)

            ago = (self.now - (c.time or self.now)).as_interval()
            try:
                percentUnderHigh = (
                    ((usePrice - c.high) / c.high) * 100 if usePrice <= c.high else 0
                )

                percentUpFromLow = (
                    ((usePrice - c.low) / c.low) * 100 if usePrice >= c.low else 0
                )

                percentUpFromClose = (
                    ((usePrice - c.close) / c.close) * 100 if c.close else 0
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

            if np.isnan(c.bidSize):
                b_s = f"{'X':>6}"
            elif 0 < c.bidSize < 1:
                b_s = f"{c.bidSize:>6.4f}"
            elif c.bidSize < 100_000:
                b_s = f"{int(c.bidSize):>6,}"
            else:
                b_s = f"{c.bidSize // 1000:>5}k"

            if np.isnan(c.askSize):
                a_s = f"{'X':>6}"
            elif 0 < c.askSize < 1:
                a_s = f"{c.askSize:>6.4f}"
            elif c.askSize < 100_000 or np.isnan(c.askSize):
                a_s = f"{int(c.askSize):>6,}"
            else:
                a_s = f"{c.askSize // 1000:>5}k"

            # use different print logic if this is an option contract or spread
            bigboi = (len(c.contract.localSymbol) > 10) or c.contract.comboLegs

            if bigboi:
                # Could use this too, but it only updates every couple seconds instead
                # of truly live with each new bid/ask update.
                # if c.modelGreeks:
                #     mark = c.modelGreeks.optPrice

                if c.bid and c.bidSize and c.ask and c.askSize:
                    mark = round((c.bid + c.ask) / 2, 2)
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
                    mark = round((c.bid + c.ask) / 2, 2) if c.bid > 0 else 0

                # For options, instead of using percent difference between
                # prices, we use percent return over the low/close instead.
                # e.g. if low is 0.05 and current is 0.50, we want to report
                #      a 900% multiple, not a 163% difference between the
                #      two numbers as we would report for normal stock price changes.
                # Also note: we use 'mark' here because after hours, IBKR reports
                # the previous day open price as the current price, which clearly
                # isn't correct since it ignores the entire most recent day.
                bighigh = (
                    ((mark / c.high if c.high else 1) - 1) * 100
                    if mark <= c.high
                    else 0
                )

                # only report low if current mark estimate is ABOVE the registered
                # low for the day, else we report it as currently trading AT the low
                # for the day instead of potentially BELOW the low for the day.
                biglow = (
                    ((mark / c.low if c.low else 1) - 1) * 100 if mark >= c.low else 0
                )

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
                    for x in c.contract.comboLegs:
                        contract = self.conIdCache[x.conId]
                        rns.append(
                            f"{x.action[0]} {x.ratio:2} {contract.localSymbol or contract.symbol}"
                        )

                    rowName = "\n".join(rns)
                    return " ".join(
                        [
                            rowName,
                            f"{fmtPriceOpt(mark):>6} ± {fmtPriceOpt(c.ask - mark):<6}",
                            f" {fmtPriceOpt(c.bid):>} x {b_s}   {fmtPriceOpt(c.ask):>} x {a_s} ",
                            "HALTED!" if c.halted > 0 else "",
                        ]
                    )
                else:
                    rowName = f"{c.contract.localSymbol or c.contract.symbol:<9}:"

                    # Also generate a nice split to reconstruct...
                    sym, y, m, d, pc, price = re.match(
                        "(\w+)\s*(\d\d)(\d\d)(\d\d)([PC])(\d\d\d\d\d\d\d\d)", rowName
                    ).groups()
                    rowNice = f"{sym:<6} {y}-{m}-{d} {pc} {int(price) / 1000:>8,.2f}"

                    # TODO: should this be fancier and decay cleaner?
                    #       we could do more accurate countdowns to actual expiration time instead of just "days"
                    when = (
                        pendulum.parse(f"20{y}-{m}-{d} 16:00", tz="US/Eastern")
                        - self.now
                    ).days

                    # TODO: why isn't this updating?
                    e100 = getEMA(ls, 100)
                    e100diff = (mark - e100) if e100 else None

                    # this may be too wide for some people? works for me.
                    # just keep shrinking your terminal font size until everything fits?
                    # currently works nicely via:
                    #   - font: Monaco
                    #   - size: 10
                    #   - terminal width: 275+ characters
                    #   - terminal height: 60+ characters
                    return " ".join(
                        [
                            rowName,
                            f"[u {fmtPricePad(und, padding=8, decimals=2)} ({underlyingStrikeDifference or -0:>7,.2f}%)]",
                            f"[iv {iv or 0:.2f}]",
                            f"[d {delta or 0:>5.2f}]",
                            f"{fmtPriceOpt(e100):>6} ({fmtPricePad(e100diff, padding=6)})",
                            f"{fmtPriceOpt(mark):>6} ±{fmtPriceOpt(c.ask - mark):<4}",
                            # f"{fmtPriceOpt(usePrice)}",
                            f"({pctBigHigh} {amtBigHigh} {fmtPriceOpt(c.high):>6})",
                            f"({pctBigLow} {amtBigLow} {fmtPriceOpt(c.low):>6})",
                            f"({pctBigClose} {amtBigClose} {fmtPriceOpt(c.close):>6})",
                            f" {fmtPriceOpt(c.bid):>6} x {b_s}   {fmtPriceOpt(c.ask):>6} x {a_s} ",
                            f"  ({str(ago):>13})  ",
                            rowNice,
                            f"({when:>3} d)",
                            "HALTED!" if c.halted > 0 else "",
                        ]
                    )

            # TODO: pre-market and after-market hours don't update the high/low values, so these are
            #       not populated during those sessions.
            #       this also means during after-hours session, the high and low are fixed to what they
            #       were during RTH and are no longer valid. Should this have a time check too?
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

            e100 = getEMA(ls, 300)
            e30 = getEMA(ls, 100)

            # for price differences we show the difference as if holding a LONG position
            # at the historical price as compared against the current price.
            # (so, if e100 is $50 but current price is $55, our difference is +5 because
            #      we'd have a +5 profit if held from the historical price.
            #      This helps align "price think" instead of showing difference from historical
            #      vs. current where "smaller historical vs. larger current" would cause negative
            #      difference which is actually a profit if it were LONG'd in the past)
            e100diff = (usePrice - e100) if e100 else None
            e30diff = (usePrice - e30) if e30 else None
            # logger.info("[{}] e100 e30: {} {} {} {}", ls, e100, e30, e100diff, e30diff)

            return (
                f"{ls:<9}: {fmtPricePad(e100)} ({fmtPricePad(e100diff, padding=6)}) {fmtPricePad(e30)} ({fmtPricePad(e30diff, padding=6)}) {fmtPricePad(usePrice)}  ({pctUndHigh} {amtUndHigh}) ({pctUpLow} {amtUpLow}) ({pctUpClose} {amtUpClose}) {fmtPricePad(c.high)}   {fmtPricePad(c.low)} <aaa bg='purple'>{fmtPricePad(c.bid)} x {b_s} {fmtPricePad(c.ask)} x {a_s}</aaa>  {fmtPricePad(c.open)} {fmtPricePad(c.close)}    ({str(ago)})"
                + ("     HALTED!" if c.halted > 0 else "")
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
                # This double symbol check is so we don't accidentially sort index ETF options
                # inside the ETF section.
                if c.secType in {"FUT", "IND"} or (
                    (c.symbol == c.localSymbol)
                    and (
                        c.symbol
                        in {
                            "SPY",
                            "QQQ",
                            "IWM",
                            "DIA",
                        }
                    )
                ):
                    priority = FUT_ORD[c.symbol] if c.symbol in FUT_ORD else 0
                    return (0, priority, c.symbol)

                # draw crypto quotes under futures quotes
                if c.secType == "CRYPTO":
                    priority = 0
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

            # RegT overnight margin can be at maximum 50% of total account value.
            # (note: does not apply to portfolion margin / SPAN accounts)
            # "TotalCashValue" will be negative if using margin, so the negative amount is the
            # current margin used. Overnight margin must be no more than half current account value,
            # but the account value includes the margin loan, so overnight margin use must be only
            # up to half the total balance value (so overnight margin must be less than the total
            # cash+equity balance itself).
            overnightDeficit = (
                0
                if self.accountStatus["TotalCashValue"] >= 0
                else (
                    self.accountStatus["TotalCashValue"]
                    + self.accountStatus["NetLiquidation"]
                )
            )

            onc = ""
            if overnightDeficit < 0:
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

                spxbreakers = f"7%: {spxc7} ({spxcd7}; {undX(spxcd7, spxc7):.2f}%)   13%: {spxc13}  ({spxcd13}; {undX(spxcd13, spxc13):.2f}%)  20%: {spxc20} ({spxcd20}; {undX(spxcd20, spxc20):.2f}%)"

            return HTML(
                f"""{self.now}{onc} [{self.updates:,}]                {spxbreakers}\n"""
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

        self.dispatch = lang.Dispatch()
        pygame.mixer.init()

        # TODO: could probably just be: pathlib.Path(__file__).parent
        pygame.mixer.music.load(pathlib.Path(__file__).parent / "CANYON.MID")

        contracts = [Stock(sym, "SMART", "USD") for sym in stocks]
        contracts += futures
        contracts += idxs

        # flip to enable/disable verbose ib_insync library logging
        if False:
            import logging

            ib_insync.util.logToConsole(logging.INFO)

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
        # self.ib.pendingTickersEvent += self.tickersUpdate

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

            # resubscribe to active quotes
            # remove all quotes and re-subscribe to the current quote state
            logger.info("[quotes] Restoring quote state...")
            self.quoteState.clear()

            await self.dispatch.runop("qrestore", "global", self.opstate)
            logger.info("[quotes] All quotes subscribed!")

            # We used to think this needed to be called before each new market data request, but
            # apparently it works fine now only set once up front?
            # Tell IBKR API to return "last known good quote" if outside
            # of regular market hours instead of giving us bad data.
            self.ib.reqMarketDataType(2)

            for contract in contracts:
                # Additional details can be requested:
                # https://ib-insync.readthedocs.io/api.html#ib_insync.ib.IB.reqMktData
                # https://interactivebrokers.github.io/tws-api/tick_types.html
                # By default, only common fields are populated (so things like 13/26/52 week
                # highs and lows aren't created unless requested via tick set 165, etc)
                # Also can subscribe to live news feed per symbol with tick 292 (news result
                # returned via tickNewsEvent callback, we think)

                tickFields = tickFieldsForContract(contract)
                self.quoteState[contract.symbol] = self.ib.reqMktData(
                    contract, tickFields
                )

                self.quoteContracts[contract.symbol] = contract

        async def reconnect():
            # don't reconnect if an exit is requested
            if self.exiting:
                return

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

                    await self.ib.connectAsync(
                        self.host,
                        self.port,
                        clientId=int(os.getenv("ICLI_CLIENT_ID", 0)),
                        readonly=False,
                        account=self.accountId,
                    )

                    logger.info(
                        "Connected! Current Request ID: {}", self.ib.client._reqIdSeq
                    )

                    self.connected = True

                    self.ib.reqNewsBulletins(True)

                    await requestMarketData()

                    # reset cached states on reconnect so we don't show stale data
                    self.summary.clear()
                    self.position.clear()
                    self.order.clear()
                    self.pnlSingle.clear()

                    # request live updates (well, once per second) of account and position values
                    self.ib.reqPnL(self.accountId)

                    # Subscribe to realtime PnL updates for all positions in account
                    # Note: these are updated once per second per position! nice.
                    # TODO: add this to the account order/filling notifications too.
                    for p in self.ib.portfolio():
                        self.pnlSingle[p.contract.conId] = self.ib.reqPnLSingle(
                            self.accountId, "", p.contract.conId
                        )

                    if False:
                        # Optionally we can subscribe to live bars for futures if we
                        # want to run a real time futures price algo too.
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

                        self.liveBars = {
                            c.symbol: self.ib.reqRealTimeBars(c, 5, "TRADES", False)
                            for c in lookupBars
                        }

                    # run some startup accounting subscriptions concurrently
                    await asyncio.gather(
                        self.ib.reqAccountSummaryAsync(),  # self.ib.reqPnLAsync()
                    )

                    break
                except (
                    ConnectionRefusedError,
                    asyncio.exceptions.TimeoutError,
                    asyncio.exceptions.CancelledError,
                ):
                    # Don't print exception for just a connection error
                    logger.error("Failed to connect to IB Gateway, trying again...")
                except:
                    # Do print exception for any errors not related to connection handling.
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

        set_title(f"{self.levelName().title()} Trader")
        self.ib.disconnectedEvent += lambda: asyncio.create_task(reconnect())

        session = PromptSession(
            history=ThreadedHistory(
                FileHistory(
                    os.path.expanduser(f"~/.tplatcli_ibkr_history.{self.levelName()}")
                )
            ),
            auto_suggest=AutoSuggestFromHistory(),
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
                    enable_history_search=True,
                    bottom_toolbar=self.bottomToolbar,
                    # refresh_interval=3,
                    # mouse_support=True,
                    # completer=completer, # <-- causes not to be full screen due to additional dropdown space
                    complete_in_thread=True,
                    complete_while_typing=True,
                )

                # Attempt to run the command(s) submitted into the prompt
                # (allow pasting multiple newline commands)
                for ccmd in text1.strip().split("\n"):
                    cmd, *rest = ccmd.split(" ", 1)
                    with Timer(cmd):
                        try:
                            result = await self.dispatch.runop(
                                cmd, rest[0] if rest else None, self.opstate
                            )
                        except:
                            logger.exception("sorry, what now?")

                continue

                if text1.startswith("rcheck "):
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

                    bid, ask, multiplier = self.currentQuote(sym)
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
                        qualified, qty=qty, isLong=isLong, ask=ask, stopPct=percentStop
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
                        await asyncio.sleep(1)
                        pass

    def stop(self):
        self.ib.disconnect()

    async def setup(self):
        pass
