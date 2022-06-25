from dataclasses import dataclass, field
from typing import *
import enum

from ib_insync import Bag, Contract

import math
import bisect
import datetime
from collections import Counter, defaultdict

import mutil.dispatch
from mutil.dispatch import DArg
from mutil.numeric import fmtPrice
from mutil.frame import printFrame

import pandas as pd
import numpy as np

from loguru import logger
from icli.helpers import *
import icli.orders as orders
import tradeapis.buylang as buylang

import asyncio
import aiohttp
import pygame

import prettyprinter as pp

pp.install_extras(["dataclasses"], warn_on_error=False)

Symbol = str

# The choices map nice user strings to the lookup map in orders.order(orderType)
# for dynamically returning an order instance based on string name...
ORDER_TYPE_Q = Q(
    "Order Type",
    choices=[
        Choice("Limit", "LMT"),
        Choice("Adaptive Fast", "LMT + ADAPTIVE + FAST"),
        Choice("Adaptive Slow", "LMT + ADAPTIVE + SLOW"),
        Choice("Peg Primary (RELATIVE)", "REL"),
        Choice("MidPrice", "MIDPRICE"),
        Choice("Market", "MKT"),
        Choice("Adaptive Fast Market", "MKT + ADAPTIVE + FAST"),
        Choice("Adaptive Slow Market", "MKT + ADAPTIVE + SLOW"),
        Choice("Market on Open (MOO)", "MOO"),
        Choice("Market on Close (MOC)", "MOC"),
    ],
)

# Also map for user typing shorthand on command line order entry.
# Values aliases are allowed for easier memory/typing support.
ALGOMAP = dict(
    LMT="LMT",
    LIM="LMT",
    LIMIT="LMT",
    AF="LMT + ADAPTIVE + FAST",
    AS="LMT + ADAPTIVE + SLOW",
    MID="MIDPRICE",
    MP="MIDPRICE",
    REL="REL",
    AFM="MKT + ADAPTIVE + FAST",
    ASM="MKT + ADAPTIVE + SLOW",
    AMF="MKT + ADAPTIVE + FAST",
    AMS="MKT + ADAPTIVE + SLOW",
    MOO="MOO",
    MOC="MOC",
)


def lookupKey(contract):
    """Given a contract, return something we can use as a lookup key.

    Needs some tricks here because spreads don't have a built-in
    one dimensional representation."""

    # exclude COMBO/BAG orders from local symbol replacement because
    # those show the underlying symbol as localSymbol and it doesn't
    # look like a spread/bag/combo.
    if contract.localSymbol and not contract.tradingClass == "COMB":
        return contract.localSymbol.replace(" ", "")

    # else, if a regular symbol but DOESN'T have a .localSymbol (means
    # we added the quote from a contract without first qualifying it,
    # which works, it's just missing extra details)
    if contract.symbol and not contract.comboLegs:
        return contract.symbol

    # else, is spread so need to make something new...
    return tuple(
        x.tuple()
        for x in sorted(contract.comboLegs, key=lambda x: (x.ratio, x.action, x.conId))
    )


@dataclass
class IOp(mutil.dispatch.Op):
    """Common base class for all operations.

    Just lets us have a single point for common settings across all ops."""

    def __post_init__(self):
        # for ease of use, populate state IB into our own instance
        assert self.state
        self.ib = self.state.ib
        self.cache = self.state.cache


@dataclass
class IOpQQuote(IOp):
    """Quick Quote: Run a temporary quote request then print results when volatility is populated."""

    def argmap(self) -> list[DArg]:
        return [DArg("*symbols")]

    async def run(self):
        if not self.symbols:
            logger.error("No symbols requested?")
            return

        contracts = [contractForName(sym) for sym in self.symbols]

        await self.state.qualify(*contracts)

        if not all(c.conId for c in contracts):
            logger.error("Not all contracts reported successful lookup!")
            logger.error(contracts)
            return

        # BROKEN IBKR WORKAROUND:
        # For IV/HV calculations, IBKR seems to have a server-side background
        # process to generate them on-demand, but if they don't exist
        # when your quote is requested, the fields will never be
        # populated.
        # Requesting quotes outside of RTH often takes 5-30 seconds to
        # deliver IV/HV calculations (if they get populated at all).
        # So we have to REQUEST the data, hope IBKR spawns its
        # background calculations, CANCEL the request, then RE-REQUEST
        # the data and hopefully it is populated the second time.
        # Also, the first cancel must be delayed enough to trigger their
        # background calculations or else they still won't populate
        # in the future.
        # Also also, the second request must be soon after the first
        # (but not *TOO* soon!) or else their background cache will
        # clear their background-created cached value and the entire
        # double-request cycle needs to start again.
        totalTry = 0
        while True:
            tickers = []
            logger.info(
                "Requesting tickers for {}",
                ", ".join(
                    [c.localSymbol.replace(" ", "") or c.symbol for c in contracts]
                ),
            )
            for contract in contracts:
                # Request quotes with metadata fields populated
                # (note: metadata is only populated using "live" endpoints,
                #  so we can't use the self-canceling "11 second snapshot" parameter)
                tickers.append(
                    self.ib.reqMktData(
                        contract,
                        tickFieldsForContract(contract),
                    )
                )

            # Loop over quote results until they have all been reported
            success = False
            for i in range(0, 3):
                ivhv = [(t.impliedVolatility, t.histVolatility) for t in tickers]

                # if any iv/hv are still nan, don't stop yet.
                if np.isnan(ivhv).any() and totalTry < 10:
                    logger.warning("Still missing fields...")
                    totalTry += 1
                else:
                    if totalTry >= 10:
                        logger.warning("Quote never finished. Final state:")

                    # if all iv and hv are populated, stop!
                    success = True

                    df = pd.DataFrame(tickers)

                    # extract contract data from nested object pandas would otherwise
                    # just convert to a blob of json text.
                    contractframe = pd.DataFrame([t.contract for t in tickers])

                    if contractframe.empty:
                        logger.error("No result!")
                        continue

                    contractframe = contractframe["symbol secType conId".split()]

                    # NB: 'halted' statuses are:
                    # -1 Halted status not available.
                    # 0 Not halted.
                    # 1 General halt. regulatory reasons.
                    # 2 Volatility halt.
                    df = df[
                        """bid bidSize
                           ask askSize
                           last lastSize
                           volume open high low close
                           halted shortableShares
                           histVolatility impliedVolatility""".split()
                    ]

                    # attach inner name data to data rows since it's a nested field thing
                    # this 'concat' works because the row index ids match across the contracts
                    # and the regular ticks we extracted.
                    df = pd.concat([contractframe, df], axis=1)

                    printFrame(df)
                    break

                await asyncio.sleep(2)

            if success:
                break

            for contract in contracts:
                self.ib.cancelMktData(contract)

            # try again...
            await asyncio.sleep(0.333)

        # all done!
        for contract in contracts:
            self.ib.cancelMktData(contract)


@dataclass
class IOpPositionEvict(IOp):
    """Evict a position using automatic MIDPRICE sell order for equity or ADAPTIVE FAST for options and futures.

    Note: the symbol name accepts '*' for wildcards!

    Also note: for futures, the actual symbol is the month expiration like "MESU2" and not just "MES",
               so to evict futures you want to evict MES* and not just MES or /MES.
    """

    def argmap(self):
        return [
            DArg("sym"),
            DArg(
                "qty",
                int,
                lambda x: x != 0 and x >= -1,
                "qty is the exact quantity to evict (or -1 to evict entire position)",
            ),
            DArg(
                "delta",
                float,
                lambda x: 0 <= x <= 1,
                "only evict matching contracts with current delta >= X (not used if symbol isn't an option). deltas are positive for all contracts in this case (so asking for 0.80 will evict calls with delta >= 0.80 and puts with delta <= -0.80)",
            ),
        ]

    async def run(self):
        contracts = self.state.contractsForPosition(
            self.sym, None if self.qty == -1 else self.qty
        )

        if not contracts:
            logger.error("No contracts found for: {}", self.sym)
            return None

        for contract, qty, price in contracts:
            if self.delta:
                # if asking for a delta eviction, check current quote...
                quotesym = contract.localSymbol.replace(" ", "")

                # verify quote is loaded...
                await self.runoplive(
                    "add",
                    f'"{quotesym}"',
                )

                # check delta...
                while not (
                    thebigd := self.state.quoteState[quotesym].modelGreeks.delta
                ):
                    # takes a couple moments for the greeks feed to populate on initial quote...
                    await asyncio.sleep(0.003)

                # skip placing this contract order if the delta is below the user requested threshold.
                # (i.e. equivalent to "only evict if self.delta >= abs(contract delta)")
                if abs(thebigd) < self.delta:
                    continue

            await self.state.qualify(contract)

            # set price floor to 3% below current live price for
            # the midprice order floor.
            if qty < 0:
                # if position IS SHORT, this is a BUY so we need a HIGHER CAP
                limit = round(price * 1.03, 2)

                if isinstance(contract, Option):
                    # options have deeper exit floor criteria because their ranges can be wider.
                    # the IBKR algo is "adaptive fast" so it should *try* to pick a good value in
                    # the spread without immediately going to market, but ymmv.
                    limit = round(price * 1.25, 2)
            else:
                # else, position IS LONG, this is a SELL, so we need a LOWER CAP
                limit = round(price / 1.03, 2)

                if isinstance(contract, Option):
                    # options have deeper exit floor criteria because their ranges can be wider.
                    # the IBKR algo is "adaptive fast" so it should *try* to pick a good value in
                    # the spread without immediately going to market, but ymmv.
                    limit = round(price / 1.25, 2)

            algo = "MIDPRICE"

            if len(contract.localSymbol) > 10 or isinstance(contract, Future):
                algo = "LMT + ADAPTIVE + FAST"

            # if limit price rounded down to zero, just do a market order
            if not limit:
                algo = "MKT + ADAPTIVE + FAST"

            logger.info(
                "[{}] [{}] Submitting...",
                self.sym,
                (contract.localSymbol, qty, price, limit),
            )

            # using MIDPRICE for equity-like things and ADAPTIVE for option-like things.
            # TODO: review this and see if maybe it should be hooked up to just price tracking algo?

            # TODO: when trades complete, have trade event send "trade done" event to listeners for
            #       next chained action (e.g. EVICT SPXW* -1 0.78 ... THEN BUY MORE ... FAST SPX P {price} 0)
            trade = await self.state.placeOrderForContract(
                contract.localSymbol,  # TODO: may be unnecessary since 'contract' has symbols too...
                # True==BUY if currently short so _BUY_ TO CLOSE, False==SELL if currently long so _SELL_ TO CLOSE
                qty < 0,
                contract,
                abs(qty),
                limit,
                algo,
            )


@dataclass
class IOpCash(IOp):
    def argmap(self):
        return []

    async def run(self):
        result = {
            "Avail Full": [
                self.state.accountStatus["AvailableFunds"],
                self.state.accountStatus["AvailableFunds"] * 2,
                self.state.accountStatus["AvailableFunds"] * 4,
            ],
            "Avail Buffer": [
                self.state.accountStatus["AvailableFunds"] / 1.10,
                self.state.accountStatus["AvailableFunds"] * 2 / 1.10,
                self.state.accountStatus["AvailableFunds"] * 4 / 1.10,
            ],
            "Net Full": [
                self.state.accountStatus["NetLiquidation"],
                self.state.accountStatus["NetLiquidation"] * 2,
                self.state.accountStatus["NetLiquidation"] * 4,
            ],
            "Net Buffer": [
                self.state.accountStatus["NetLiquidation"] / 1.10,
                self.state.accountStatus["NetLiquidation"] * 2 / 1.10,
                self.state.accountStatus["NetLiquidation"] * 4 / 1.10,
            ],
        }

        printFrame(
            pd.DataFrame.from_dict(
                {k: [f"${v:,.2f}" for v in vv] for k, vv in result.items()},
                orient="index",
                columns=["Cash", "Overnight", "Day"],
            )
        )


@dataclass
class IOpAlias(IOp):
    def argmap(self):
        return [DArg("cmd"), DArg("*args")]

    async def run(self):
        # TODO: allow aliases to read arguments and do calculations internally
        # TODO: should this just be an external parser language too?
        aliases = {
            "buy-spx": {"async": ["fast spx c :1 0 :2*"]},
            "sell-spx": {"async": ["evict SPXW* -1"]},
            "clear-quotes": {"async": ["qremove blahblah SPXW*"]},
            # TODO: if RTH, use 4x, if PM or AH use 2x
            "buy-screen": {
                "var": {"spend": self.state.accountStatus["AvailableFunds"] * 4 / 1.10},
                "spend-per-order": {"calc": ":spend / ::asynclength"},
                "async": [
                    f"buy buy {sym} q 0 p :spend-per-order a AF"
                    for sym in {
                        "AAPL",
                        "SHOP",
                        "FB",
                        "NVDA",
                        "AMD",
                        "MSFT",
                        "TWLO",
                        "ETSY",
                        "ROKU",
                    }
                ],
            },
        }

        if self.cmd not in aliases:
            logger.error("[alias {}] Not found?", self.cmd)
            logger.error("Available aliases: {}", sorted(aliases.keys()))
            return None

        cmd = aliases[self.cmd]
        logger.info("[alias {}] Running: {}", self.cmd, cmd)

        # TODO: we could make a simpler run wrapper for "run command string" instead of
        #       always breaking out the command-vs-argument strings manually.
        return await asyncio.gather(
            *[
                self.runoplive(
                    cmd.split()[0],
                    " ".join(cmd.split()[1:]),
                )
                for cmd in cmd["async"]
            ]
        )


@dataclass
class IOpDepth(IOp):
    def argmap(self):
        return [
            DArg("sym"),
            DArg(
                "count",
                convert=int,
                verify=lambda x: 0 < x < 300,
                desc="depth checking iterations should be more than zero and less than a lot",
            ),
        ]

    async def run(self):
        try:
            (contract,) = await self.state.qualify(contractForName(self.sym))
        except:
            logger.error("No contract found for: {}", self.sym)
            return

        # logger.info("Available depth: {}", await self.ib.reqMktDepthExchangesAsync())

        self.depthState = {}
        useSmart = True
        self.depthState[contract] = self.ib.reqMktDepth(
            contract, numRows=40, isSmartDepth=useSmart
        )

        # now we read lists of ticker.domBids and ticker.domAsks for the depths
        # (each having .price, .size, .marketMaker)
        for i in range(0, self.count):
            t = self.depthState[contract]

            # loop for up to a second until bids or asks are populated
            for j in range(0, 100):
                if not (t.domBids or t.domAsks):
                    await asyncio.sleep(0.001)

            if not (t.domBids or t.domAsks):
                logger.warning(
                    "[{}] Depth not populated. Failing depth {}.",
                    contract.localSymbol,
                    i,
                )

            if t.domBids or t.domAsks:
                if False:
                    fmt = {
                        "Bids": pd.DataFrame(t.domBids),
                        "Asks": pd.DataFrame(t.domAsks),
                    }
                    printFrame(
                        pd.concat(fmt, axis=1).fillna(""),
                        f"{contract.symbol} by Market",
                    )

                # Also show grouped by price with sizes summed and markets joined
                # These frames are a bit of a mess:
                # - Create frame
                # - Group frame by price so same prices use one row
                # - Add all sizes for the same price, and concatenate marketMaker cols
                # - 'convert_dtypes()' so any collapsed rows become None instead of NaN
                #   (aggregation sum of int + NaN = float, but we want int, so we use
                #    int + None = int to stop decimals from appearing in the size sums)
                # - re-sort by price based on side
                #   - bids: high to low
                #   - asks: low to high
                # - Re-index the frame by current sorted positions so the concat joins correctly.
                #   - 'drop=True' means don't add a new column with the previous index value

                # condition dataframe reorganization on the input list existing.
                # for some smaller symbols, bids or asks may not get returned
                # by the flaky ibkr depth APIs
                if t.domBids:
                    fixedBids = (
                        pd.DataFrame(t.domBids)
                        .groupby("price", as_index=False)
                        .agg({"size": sum, "marketMaker": list})
                        .convert_dtypes()
                        .sort_values(by=["price"], ascending=False)
                        .reset_index(drop=True)
                    )

                    # format floats as currency strings with proper cent padding
                    fixedBids["price"] = fixedBids["price"].apply(lambda x: f"{x:,.2f}")
                    fixedBids["marketMaker"] = sorted(fixedBids["marketMaker"])

                else:
                    fixedBids = pd.DataFrame()

                if t.domAsks:
                    fixedAsks = (
                        pd.DataFrame(t.domAsks)
                        .groupby("price", as_index=False)
                        .agg({"size": sum, "marketMaker": list})
                        .convert_dtypes()
                        .sort_values(by=["price"], ascending=True)
                        .reset_index(drop=True)
                    )

                    fixedAsks["price"] = fixedAsks["price"].apply(lambda x: f"{x:,.2f}")
                    fixedAsks["marketMaker"] = sorted(fixedAsks["marketMaker"])
                else:
                    fixedAsks = pd.DataFrame()

                fmtJoined = {"Bids": fixedBids, "Asks": fixedAsks}

                # Create an order book with high bids and low asks first.
                # Note: due to the aggregations above, the bids and asks
                #       may have different row counts. Extra rows will be
                #       marked as <NA> by pandas (and we can't fill them
                #       as blank because the cols have been coerced to
                #       specific data types via 'convert_dtypes()')
                both = pd.concat(fmtJoined, axis=1)
                printFrame(
                    both,
                    f"{contract.symbol} :: {contract.localSymbol} Grouped by Price",
                )

            # Note: the 't.domTicks' field is just the "update feed"
            #       which ib_insync merges into domBids/domAsks
            #       automatically, so we don't need to care about
            #       the values inside t.domTicks

            if i < self.count - 1:
                await asyncio.sleep(3)

        self.ib.cancelMktDepth(contract, isSmartDepth=useSmart)
        del self.depthState[contract]


@dataclass
class IOpRID(IOp):
    """Retrieve ib_insync request ID and server Next Request ID"""

    def argmap(self):
        # rid has no args!
        return []

    async def run(self):
        logger.info("CLI Request ID: {}", self.ib.client._reqIdSeq)
        logger.info(
            "Server Next Request ID: {} (see server log)", self.ib.client.reqIds(0)
        )


@dataclass
class IOpOrderModify(IOp):
    """Modify an existing order using interactive prompts."""

    def argmap(self):
        # No args, we just use interactive prompts for now
        return []

    async def run(self):
        # "openTrades" include the contract, order, and order status.
        # "openOrders" only includes the order objects with no contract or status.
        ords = self.ib.openTrades()
        # logger.info("current orderS: {}", pp.pformat(ords))
        promptOrder = [
            Q(
                "Current Order",
                choices=[
                    Choice(
                        f"{o.order.action:<4} {o.order.totalQuantity:<6} {o.contract.localSymbol or o.contract.symbol:<21} {o.order.orderType} {o.order.tif} lmt:${fmtPrice(o.order.lmtPrice):<7} aux:${fmtPrice(o.order.auxPrice):<7}",
                        o,
                    )
                    for o in sorted(ords, key=tradeOrderCmp)
                ],
            ),
            Q("New Limit Price"),
            Q("New Stop Price"),
            Q("New Quantity"),
        ]

        pord = await self.state.qask(promptOrder)
        try:
            trade = pord["Current Order"]
            lmt = pord["New Limit Price"]
            stop = pord["New Stop Price"]
            qty = pord["New Quantity"]

            contract = trade.contract
            ordr = trade.order

            if not (lmt or stop or qty):
                # User didn't provide new data, so stop processing
                return None

            if lmt:
                ordr.lmtPrice = float(lmt)

            if stop:
                ordr.auxPrice = float(stop)

            if qty:
                ordr.totalQuantity = float(qty)
        except:
            return None

        trade = self.ib.placeOrder(contract, ordr)
        logger.info("Updated: {}", pp.pformat(trade))


@dataclass
class IOpOrder(IOp):
    """Quick order entry with full order described on command line."""

    def argmap(self):
        # TODO: write a parser for this language instead of requiring fixed orders for each parameter?
        # allow symbol on command line, optionally
        # BUY IWM TOTAL 50000 ALGO LIMIT/LIM/LMT AF AS REL MP AMF AMS MOO MOC
        return [
            DArg("symbol"),
            DArg("bs", verify=lambda x: x.lower() in {"b", "s", "buy", "sell"}),
            DArg("t", verify=lambda x: x.lower() in {"t", "total"}),
            DArg("total", convert=float, verify=lambda x: x > 0),
            DArg("a", verify=lambda x: x.lower() in {"a", "algo"}),
            DArg(
                "algo",
                convert=lambda x: x.upper(),
                verify=lambda x: x in ALGOMAP.keys(),
                errmsg=f"Available algos: {pp.pformat(ALGOMAP)}",
            ),
        ]

    async def run(self):
        if " " in self.symbol:
            # is spread, so do bag
            isSpread = True
            orderReq = self.state.ol.parse(self.symbol)
            contract = await self.state.bagForSpread(orderReq)
        else:
            # else, is symbol
            isSpread = False
            contract = contractForName(self.symbol)

        if contract is None:
            logger.error("Not submitting order because contract can't be formatted!")
            return None

        if not isSpread:
            # spreads are qualified when they are initially populated
            await self.state.qualify(contract)

        # B BUY is Long
        # S SELL is Short
        isLong = self.bs.lower().startswith("b")

        # send the order to IBKR
        # Note: negative quantity is parsed as a WHOLE DOLLAR AMOUNT to use,
        # then limit price is irrelevant since it runs a midpoint order.
        am = ALGOMAP[self.algo]
        placed = await self.state.placeOrderForContract(
            self.symbol,
            isLong,
            contract,
            # "negative quantity" means use as TOTAL PRICE
            -self.total,
            # also since we are buying by AMOUNT, we don't specify a
            # limit price since it will be calculated automatically.
            0,
            am,
        )

        if not placed:
            logger.error("[{}] Order can't continue!", self.symbol)
            return

        # if this is a market order, don't run the algo loop
        if {"MOO", "MOC", "MKT"} & set(am.split()):
            logger.warning("Not running price algo because this is a market order...")
            return

        order, trade = placed

        quoteKey = lookupKey(contract)

        checkedTimes = 0
        # while (unfilled quantity) AND (order NOT canceled or rejected or broken)
        while (rem := trade.orderStatus.remaining) > 0 or (
            "Pending" in trade.orderStatus.status
        ):
            if ("Cancel" in trade.orderStatus.status) or (
                trade.orderStatus.status in {"Filled", "Inactive"}
            ):
                logger.error(
                    "[{} :: {}] Order was canceled or rejected! Status: {}",
                    trade.orderStatus.status,
                    trade.contract.localSymbol,
                    pp.pformat(trade.orderStatus),
                )
                return

            if rem == 0:
                logger.warning(
                    "Quantity Remaining is zero, but status is Pending. Waiting for update..."
                )
                # sleep 75 ms and check again
                await asyncio.sleep(0.075)
                continue

            logger.info("Quantity remaining: {}", rem)
            checkedTimes += 1

            # if this is the first check after the order was placed, don't
            # run the algo (i.e. give the original limit price a chance to work)
            if checkedTimes == 1:
                logger.info("Skipping adjust so original limit has a chance to fill...")
                continue

            # get current qty/value of trade both remaining and already executed
            (
                remainingAmount,
                totalAmount,
                currentPrice,
                currentQty,
            ) = self.state.amountForTrade(trade)

            # get current quote for order
            bidask = self.state.currentQuote(quoteKey)
            if bidask:
                logger.info("Adjusting price for more aggressive fills...")
                bid, ask, multiplier = bidask
                if isLong:
                    # if is buy, chase the ask
                    newPrice = round((currentPrice + ask) / 2, 2)

                    # reduce qty to remain in expected total spend constraint
                    newQty = totalAmount / newPrice

                    # only crypto supports fractional values over the API,
                    # so all non-crypto contracts get floor'd
                    if not isinstance(trade.contract, Crypto):
                        newQty = math.floor(newQty)
                else:
                    # else if is sell, chase the bid
                    newPrice = round((currentPrice + bid) / 2, 2)
                    newQty = currentQty  # don't change quantities on shorts / sells
                    # TODO: this needs to be aware of CLOSING instead of OPEN SHORT.
                    # i.e. on OPENING orders we can grow/shrink qty, but on CLOSING
                    # we DO NOT want to shrink or grow our qty.

                logger.info(
                    "Price changing from {} to {} ({})",
                    currentPrice,
                    newPrice,
                    (newPrice - currentPrice),
                )
                logger.info(
                    "Qty changing from {} to {} ({})",
                    currentQty,
                    newQty,
                    (newQty - currentQty),
                )
                logger.info("Submitting order update...")
                order.lmtPrice = newPrice
                order.totalQuantity = newQty
                self.ib.placeOrder(contract, order)

            waitDuration = 3
            logger.info(
                "[{} :: {}] Waiting for {} seconds to check for new executions...",
                trade.orderStatus.orderId,
                checkedTimes,
                waitDuration,
            )

            try:
                await asyncio.sleep(waitDuration)
            except:
                # catches CTRL-C during sleep
                logger.warning(
                    "[{}] User canceled automated limit updates! Order still live.",
                    trade.orderStatus.orderId,
                )
                break


@dataclass
class IOpOrderFast(IOp):
    """Place a momentum order for scalping using multiple strikes and active price tracking.

    For a 'symbol' at total dollar spend of 'amount' and 'direction'.

    For ATR calcuation, requires a custom endpoint running with a REST API capable of
    returning multiple ATR periods for any symbol going back multiple timeframes.

    Maximum strikes attempted are based on the recent ATR for the underlying,
    so we don't try to grab a +$50 OTM strike with recent historical movement
    was only $2.50. (TODO: if running against, 2-4 week out chains allow higher
    than ATR maximum because the vol will smile).

    The buying algo is:
        - use current underlying bid/ask midpoint (requires live quote)
        - set price cap to maximum of 3 day or 20 day ATR (requires external API)
        - buy 1 ATM strike (requires fetching near-term chain for underlying)
        - for each ATM strike, buy next OTM strike up to price cap
        - if funds remaining, repeat ATM->OTM ladder
        - if funds remaining, but can't afford ATM anymore, keep trying
          the OTM ladder to spend remaining funds until exhausted.
    """

    def argmap(self):
        return [
            DArg(
                "symbol",
                convert=lambda x: x.upper(),
                verify=lambda x: " " not in x,
                desc="Underlying for contracts",
            ),
            DArg(
                "direction",
                convert=lambda x: x.upper()[0],
                verify=lambda x: x in {"P", "C"},
                desc="side to buy / trade kind to execute (puts or calls)",
            ),
            DArg("amount", convert=float, desc="Maximum dollar amount to spend"),
            DArg(
                "gaps",
                convert=int,
                desc="pick strikes N apart (e.g. use 2 for room to lock gains with butterflies)",
            ),
            DArg(
                "expirationAway",
                convert=int,
                desc="use N next expiration date (0 is NEAREST, 1 is NEXT, 2 is NEXT NEXT, ...)",
            ),
            DArg(
                "*preview",
                desc="If any other arguments present, order not placed, only order logic reported.",
            ),
        ]

    async def run(self):
        # Fetch data for our calculations:
        #   - ATR with moving 3 day window
        #   - ATR with moving 20 day window
        #   - current strikes / dates
        #   - live quote for underlying

        # TODO: may need symbol replacement for SPXW -> SPX etc?
        # TODO: make atr fetch optional and make chains fetch against
        #       an external API if configured.
        # TODO: this whole atr fetching / calculation / partitioning strikes
        #       should be an external reusable function (because it's complicating
        #       the logic flow of "calculate strike widths, place order"
        if True:
            # skip ATR for now

            # async run all URL fetches and data updates at once
            strikes = await self.runoplive(
                "chains",
                self.symbol,
            )

            if not strikes:
                logger.error("[{}] No strikes found?", self.symbol)
                return None

            # also make sure quote for the underlying is populated...
            await self.runoplive(
                "add",
                f'"{self.symbol}"',
            )
        else:
            atrReqFast = dict(cmd="atr", sym=self.symbol, avg=3, back=1)
            atrReqSlow = dict(cmd="atr", sym=self.symbol, avg=20, back=1)
            async with aiohttp.ClientSession() as cs:
                # This URL is a custom historical reporting endpoint providing various
                # technical stats on symbols given arbitrary parameters, but also requires
                # you have a DB of potentially years of daily bars for each stock.
                urls = [
                    cs.get("http://127.0.0.1:6555/", params=p)
                    for p in [atrReqFast, atrReqSlow]
                ]

                # request chains and live quote data using the language command syntax
                dataUpdates = [
                    self.runoplive(
                        "chains",
                        self.symbol,
                    ),
                    self.runoplive(
                        "add",
                        f'"{self.symbol}"',
                    ),
                ]

                # async run all URL fetches and data updates at once
                mfast_, mslow_, strikes, _ = await asyncio.gather(*(urls + dataUpdates))

                # async resolve response bodies through the JSON parser
                mfast, mslow = await asyncio.gather(
                    *[m.json() for m in [mfast_, mslow_]]
                )

        # logger.info("Got MFast, MSlow: {}", [mfast, mslow])

        # Get a valid expiration from our strikes...
        # Note: we check if it matches below, then back up if doesn't
        # TODO: allow selecting future expirations too
        # Note: this is *not* the target options expiration date, but we use the
        #       current date to bisect all chains to discover the nearest date to now.
        now = pendulum.now().in_tz("US/Eastern")

        # if after market close, use next day
        if (now.hour, now.minute) >= (16, 15):
            now = now.add(days=1)

        expTry = now.date()
        # Note: Expiration formats in the dict are *full* YYYYMMDD, not OCC YYMMDD.
        expirationFmt = f"{expTry.year}{expTry.month:02}{expTry.day:02}"

        # get nearest expiration from today (note: COULD BE TODAY!)
        # if today is NOT an expiration, this picks the CLOSEST expiration date.
        # TODO: allow selecting future expirations.
        sstrikes = sorted(strikes.keys())

        # why doesn't this work for VIX?
        # logger.info("Strikes are: {}", sstrikes)

        useExpIdx = bisect.bisect_left(sstrikes, expirationFmt)

        # this is a bit weird, but we want to find the NEXT NEAREST expiration date (even if today),
        # which is the 'useExpIdx', then we want the requested expire away, which is 'self.expirationAway'
        # distance from the currently found expiration date, so we get a slice of the expirationAway length,
        # then we take the last element which is our actually requested expiration away date.
        useExp = sstrikes[useExpIdx : useExpIdx + 1 + self.expirationAway][-1]

        assert useExp in strikes

        useChain = strikes[useExp]

        logger.info(
            "[{} :: {}] Using expiration {} (days away: {}) chain: {}",
            self.symbol,
            self.direction,
            useExp,
            int(useExp) - int(expirationFmt),
            useChain,
        )

        if False:
            maximumATR: float = 0
            prevClose: float = 0
            prevHigh: float = 0

            # just not running for now, skipping ATR limits
            for df in (mfast, mslow):
                data = df["data"]
                date: str = list(data.keys())[-1]
                fordate = data[date]
                atr: float = fordate["atr"]

                if atr > maximumATR:
                    maximumATR = atr
                    prevClose = fordate["close"]
                    prevHigh = fordate["high"]

        # quote is already live in quote state...

        # HACK TO GET AROUND NOT HAVING INDEX QUOTES
        useQuoteSymbol = self.symbol
        if useQuoteSymbol in {"SPX", "SPXW"}:
            # only works if you're subscribing to Index(SPX, CBOE) things. can also replace with "ES"
            # if you want to base of the current /ES quote...
            # TODO: should we do something like "overnight and pre-market, use ES, else use live SPX?"
            useQuoteSymbol = "SPX"
        elif useQuoteSymbol in {"NDX", "NDXW"}:
            # Same problem with SPX, but NDX requires different data subscriptions, but
            # you may have /MNQ or /NQ instead which may be ~okay?
            useQuoteSymbol = "MNQ"

        quote = self.state.quoteState[useQuoteSymbol]

        # if we subscribed to new quotes, wait a little while for
        # the IBKR streaming API to start populating our quotes...
        for i in range(0, 5):
            # currentLow: float = quote.last # RETURN TO: quote.low
            # TEMPORARY HACK BECAUSE OVER WEEKEND "SPX" returns NAN for LAST and LOW. SIGH.
            currentLow: float = quote.last if quote.last == quote.last else quote.close

            # note: this can break during non-market hours when
            #       IBKR returns random bad values for current
            #       prices (also after market hours their bid/ask
            #       no longer populates so we can't just make it
            #       a midpoint).
            # Though, quote.last seems more reliable than quote.marketPrice()
            # for any testing after hours/weekends/etc.
            # TODO: could also use 'underlying' value from full OCC quote, but need
            #       a baseline to get the first options quotes anyway...

            # used to look up the ATM amount here:
            currentPrice: float = (
                quote.last if quote.last == quote.last else quote.close
            )

            if any(np.isnan([currentLow, currentPrice])):
                logger.warning(
                    "[{} :: {}] [{}] Quotes not all populated ({}). Waiting for activity...",
                    self.symbol,
                    self.direction,
                    i,
                    quote.last,
                )

                await asyncio.sleep(0.140)
            else:
                logger.info(
                    "[{} :: {}] Using quote last price: {}",
                    self.symbol,
                    self.direction,
                    currentPrice,
                )
                break

        # sort a nan-free list of atr goals
        # (note: sorting here because we want to print it in the log, so
        #        below instead of max() we can also just do [-1] since we
        #        already sorted here)
        # TODO: expose this sorted list to the user so they can pick their
        #       "upper aggressive level" tolerance (low, medium, high)
        usingCalls = self.direction == "C"

        if np.isnan([currentPrice]):
            logger.error("No current price found, can't continue.")
            return None

        # boundaryPrice hack because we don't have the ATR server at the moment
        # NOTE: this gets updated for range extremes during the strike calculation loop.
        boundaryPrice = currentPrice

        if False:
            # not using ATR for now
            if usingCalls:
                # call goals sort forward with the highest range as the rightmost value
                atrGoals = [
                    x
                    for x in sorted(
                        [
                            prevClose + maximumATR,
                            prevHigh + maximumATR,
                            currentLow + maximumATR,
                        ]
                    )
                    if x == x  # drop nan values if we are missing a quote
                ]
            else:
                # put goals sort backwards with lowest range as the rightmost value
                atrGoals = [
                    x
                    for x in reversed(
                        sorted(
                            [
                                prevClose - maximumATR,
                                prevHigh - maximumATR,
                                currentLow - maximumATR,
                            ]
                        )
                    )
                    if x == x  # drop nan values if we are missing a quote
                ]

        # boundary value (the up/down we don't want to cross)
        # is always last element of the sorted goals list
        if False:
            boundaryPrice = atrGoals[-1]

        # it's possible current-day performance is better (or worse) than
        # expected ATR-calculated expected performance, so if underlying
        # is already beyond the maximum ATR calculation
        # (either above it for calls or under it for puts), then we
        # need to adjust the full boundary expectation off the current
        # price because maybe this is going to be a double ATR day.
        # (otherwise, if we don't adjust, the conditions below end
        #  up empty because it tries to interrogate an empty range
        #  because currentPrice would be higher than expected prices
        #  to buy (for calls), so we'd never generate a buyQty)
        if False:
            if usingCalls:
                if boundaryPrice <= currentPrice:
                    boundaryPrice += maximumATR
            else:
                if boundaryPrice >= currentPrice:
                    boundaryPrice -= maximumATR

        # collect strikes near the current price in gaps as requested
        # until we reach the maximum price
        # (and walk _backwards_ if doing puts!)
        firstStrikeIdx = bisect.bisect_left(useChain, currentPrice)

        # because of sorting, the bisect selects an ITM strike for
        # initial puts, but we can back it down one to start at
        # the first OTM strike (or it will be ATM if the prices
        # match exactly).
        if not usingCalls and firstStrikeIdx > 0:
            if useChain[firstStrikeIdx] > currentPrice:
                firstStrikeIdx -= 1

        buyStrikes = []

        # if puts, walk towards lower strikes instead of higher strikes.
        directionMul = 1 if usingCalls else -1

        # If asking for negative gaps (more ITM instead of more OTM) then our
        # direction is inverted from normal for the call/put strike discovery.
        # (e.g. gaps >= 0 == go MORE OTM; gaps < 0 == go MORE ITM)
        # TODO: fix direction for -0 (need go use ±1 instead of 0 for no gaps...)
        # directionMul = -directionMul if abs(self.gaps) != self.gaps else directionMul

        # the range step paraemter is the "+=" increment between
        # iterations, so step=2 returns 0, 2, 4, ...;
        # step=3 returns 0, 3, 6, 9, ...
        # but our "gaps" params is numbers BETWEEN steps, so we
        # need steps=(gaps+1) becaues the step is 'inclusive' of the
        # next result value, but our 'gaps' is fully exclusive gaps.
        while len(buyStrikes) < 3:
            # if we didn't find a boundary price, then extend it a bit more.
            # NOTE: This is still a hack because our ATR server isn't live again.
            boundaryPrice = boundaryPrice * 1.01 if usingCalls else boundaryPrice / 1.01

            # TODO: change initial position back to 0 from len(buyStrikes) when we restore
            #       proper ATR API reading.
            # TODO: fix gaps calculation when doing negative gaps
            # TODO: maybe make gaps=1 be NEXT strike so gaps=-1 is PREV strike for going more ITM?
            for i in range(
                len(buyStrikes), 100 * directionMul, (self.gaps + 1) * directionMul
            ):
                idx = firstStrikeIdx + i

                # if we walk over or under the strikes, we are done.
                if idx < 0 or idx >= len(useChain):
                    break

                strike = useChain[idx]

                logger.info(
                    "Checking strike vs. boundary: {} v {}", strike, boundaryPrice
                )

                # only place orders up to our maximum theoretical price
                # (either a static percent offset from current OR the actual High-to-Low ATR)
                if usingCalls:
                    # calls have an UPPER CAP
                    if strike > boundaryPrice:
                        break
                else:
                    # puts have a LOWER CAP
                    if strike < boundaryPrice:
                        break

                buyStrikes.append(strike)

        logger.info(
            "[{} :: {}] Selected strikes to purchase: {}",
            self.symbol,
            self.direction,
            buyStrikes,
        )

        # get quotes for strikes...
        occs = [
            f"{self.symbol}{useExp[2:]}{self.direction[0]}{int(strike * 1000):08}"
            for strike in buyStrikes
        ]

        logger.info("Adding quotes for: {}", " ".join(occs))
        await self.runoplive(
            "add",
            " ".join(occs),
        )

        buyQty = defaultdict(int)

        remaining = self.amount
        spend = 0
        skip = 0
        while remaining > 0 and skip < len(occs):
            logger.info(
                "[{} :: {}] Remaining: ${:,.2f} plan {}",
                self.symbol,
                self.direction,
                remaining,
                " ".join(f"{x}={y}" for x, y in buyQty.items()) or "[none yet]",
            )
            for idx, (strike, occ), in enumerate(
                zip(
                    buyStrikes,
                    occs,
                )
            ):
                # TODO: for VIX, expire date is N, but contract date is N+1, so we need
                #       to do calendar math for "add 1 day" to VIX symbols...

                # if weekly index option, needs the special name to check quotes because IBKR
                # changes our "SPX option weekly expire" dates into SPXW symbols internally, so
                # even though we request trades and quotes on "SPX" symbol, their .localSymbol
                # becomes "SPXW[OCC details]", etc.
                # Basically: All orders and quotes are placed with "SPX", "NDX", etc symbols,
                # but behind the scenes it changes them to the different root symbols as needed,
                # so for our quote lookup we need to re-construct the .localSymbol vs. the in-contract
                # order symbol.
                occForQuote = (
                    occ.replace("SPX", "SPXW")
                    .replace("VIX", "VIXW")
                    .replace("NDX", "NDXW")
                    .replace("RUT", "RUTW")
                )

                logger.info("Iterating: {}", occForQuote)

                ask = self.state.quoteState[occForQuote].ask * 100

                # if quote not populated, wait for it...
                try:
                    qs = self.state.quoteState[occForQuote]

                    # multipler is a string of a number because of course it is.
                    # It's likely always an integer, but why risk coercing to int when float is
                    # also fine here with our flakey price math.
                    multiplier = float(qs.contract.multiplier)

                    for i in range(0, 25):
                        # if ask is populated, skip rest of waiting
                        # Note: these asks only work for longs! if we have
                        # shot quotes with negative asks, all of this breaks.
                        # We check for (ask > 0) because when IBKR has no live
                        # information (or on init) it returns "ask -1" for a while,
                        # which blows up our calculations obviously.
                        if (not np.isnan(ask)) and ask > 0:
                            break

                        # else, ask is not populated, so try to wait for it...
                        logger.warning("Ask not populated. Waiting...")

                        await asyncio.sleep(0.075)

                        ask = qs.ask * multiplier
                    else:
                        # ask failed to populate, so use MP
                        logger.warning(
                            "Failed to populate quote, so using market price..."
                        )
                        ask = qs.marketPrice() * multiplier
                except KeyboardInterrupt:
                    logger.error(
                        "[FAST Order] Request termination received. Abandoning! No orders were placed."
                    )
                    return  # Control-C pressed. Goodbye.

                # attempt to weight lower strikes for optimal
                # delta capture while still getting gamma exposure
                # a little further out.
                # Current setting is:
                #   - first strike (closest to ATM) tries 3->2->1->0 buys
                #   - second strike (next furthest OTM) tries 2->1->0 buys
                #   - others try 1 buy per iteration
                for i in reversed(range(1, 4 if idx == 0 else 3 if idx == 1 else 2)):
                    if False:
                        logger.debug(
                            "[{}] Checking ({} * {}) + {} < {}",
                            occForQuote,
                            ask,
                            i,
                            spend,
                            self.amount,
                        )

                    if (ask * i) + spend < self.amount:
                        buyQty[occ] += i

                        spend += ask * i
                        remaining -= ask * i
                        break
                else:
                    # since we are only reducing 'remaining' on spend, we need
                    # another guaranteed terminiation condtion, so now we pick
                    # "terminate if skip is larger than the strikes we are buying"
                    skip += 1
                    # logger.debug(
                    #    "[{}] Skipping because doesn't fit remaining spend...", occ
                    # )

        logger.info(
            "[{} :: {}] Buying plan for {} contracts (est ${:,.2f}): {}",
            self.symbol,
            self.direction,
            sum(buyQty.values()),
            spend,
            " ".join(f"{x}={y}" for x, y in buyQty.items()),
        )

        if False:
            logger.info(
                "[{} :: {}] ATR ({:,.2f}) Estimates: {}",
                self.symbol,
                self.direction,
                maximumATR,
                atrGoals,
            )
            logger.info(
                "[{} :: {}] Chosen ATR: {} :: {}",
                self.symbol,
                self.direction,
                maximumATR,
                boundaryPrice,
            )

        logger.info(
            "{}[{} :: {}] Ordering ${:,.2f} of {} apart between {:,.2f} and {:,.2f} ({:,.2f})",
            "[PREVIEW] " if self.preview else "",
            self.symbol,
            self.direction,
            self.amount,
            self.gaps,
            currentPrice,
            boundaryPrice,
            abs(currentPrice - boundaryPrice),
        )

        # now use buyDict to place orders at QTY and live track them until
        # hitting the ask.

        # don't continue to order placement if this is a preview-only request
        if self.preview:
            return None

        # qualify contracts for ordering
        # don't qualify here because they are qualified in the async order runners instead
        # contracts = {occ: contractForName(occ) for occ in buyQty.keys()}
        # await self.state.qualify(*contracts.values())

        # assemble coroutines with parameters per order
        placement = []
        for occ, qty in buyQty.items():
            qs = self.state.quoteState[
                occ.replace("SPX", "SPXW")
            ]  # TODO: FIX HACK NAME CRAP
            limit = round((qs.bid + qs.ask) / 2, 2)

            # if for some reason bid/ask aren't populated, wait for them...
            while np.isnan(limit):
                await asyncio.sleep(0.075)
                limit = round((qs.bid + qs.ask) / 2, 2)

            placement.append((occ, qty, limit, "AF"))
            # placement.append(
            #     self.state.placeOrderForContract(
            #         occ, True, contracts[occ], qty, limit, "LMT + ADAPTIVE + FAST"
            #     )
            # )

        # launch all orders concurrently
        # placed = await asyncio.gather(*placement)

        # TODO: create follow-the-spread refactor so we can
        # live follow all new strike order simmultaenously.
        # (currently around line 585)

        # re-use the "BUY" self-adjusting price algo here to place orders
        # for all our calculated strikes.
        placed = await asyncio.gather(
            *[
                self.runoplive(
                    "buy",
                    # Note: we're always running weeklies but the quote system
                    #       uses different symbols for weekly index options, so
                    #       fixup any of those here.
                    # f'buy {occ.replace("SPX", "SPXW").replace("VIX", "VIXW").replace("NDX", "NDXW").replace("RUT", "RUTW")} total {qty * price} algo {algo}',
                    # Since these are per-contract limit price, we need to re-inflate the total by the multiplier again.
                    f"{occ} buy total {qty * limit * multiplier} algo {algo}",
                )
                for occ, qty, limit, algo in placement
            ]
        )


@dataclass
class IOpOrderLimit(IOp):
    """Create a new buy or sell order using limit, market, or algo orders."""

    def argmap(self):
        # allow symbol on command line, optionally
        # symbols are retrieved using just self.args instead of being direct
        # required params here.
        return []

    async def run(self):
        promptSide = [
            Q(
                "Side",
                choices=[
                    "Buy to Open",
                    "Sell to Close",
                    "Sell to Open",
                    "Buy to Close",
                    "Butterfly Lock Gains to Close",
                ],
            ),
        ]

        gotside = await self.state.qask(promptSide)

        try:
            side = gotside["Side"]
            isClose = "to Close" in side
            isSell = "Sell" in side
            isButterflyClose = "Butterfly" in side
        except:
            # user canceled prompt, so skip
            return None

        if isClose:
            port = self.ib.portfolio()
            # Choices have a custom format/title string, but the
            # return value is the PortfolioItem object which has:
            #   .position for the entire quantity in the account
            #   .contract for the contract object to use with orders

            if isSell:
                # if SELL, show only LONG positions
                # TODO: exclude current orders already fully booked!
                portchoices = [
                    Choice(strFromPositionRow(p), p)
                    for p in sorted(port, key=portSort)
                    if p.position > 0
                ]
            else:
                # if BUY, show only SHORT positions
                portchoices = [
                    Choice(strFromPositionRow(p), p)
                    for p in sorted(port, key=portSort)
                    if p.position < 0
                ]

            promptPosition = [
                Q("Symbol", choices=portchoices),
                Q("Price"),
                Q("Quantity"),
                ORDER_TYPE_Q,
            ]
        else:
            # OPENING
            # provide any arguments as pre-populated symbol by default
            promptPosition = [
                Q("Symbol", value=" ".join(self.args)),
                Q("Price"),
                Q("Quantity"),
                ORDER_TYPE_Q,
            ]

        got = await self.state.qask(promptPosition)

        if not got:
            # user canceled at position request
            return None

        # if user canceled the form, just prompt again
        try:
            # sym here is EITHER:
            #   - input string provided by user
            #   - the cached PortfolioItem if this is a Close order on current position
            symInput = got["Symbol"]

            contract = None
            portItem = None  # portfolioItem
            # if sym is a CONTRACT, make it explicit and also
            # retain the symbol name independently
            if isinstance(symInput, str):
                sym = symInput
            else:
                portItem = symInput
                contract: Contract = portItem.contract
                sym: str = contract.symbol

            qty = float(got["Quantity"])
            price = float(got["Price"])
            isLong = gotside["Side"].startswith("Buy")
            orderType = got["Order Type"]
        except:
            # logger.exception("Failed to get field?")
            logger.exception("Order creation canceled by user")
            return None

        # if order is To Close, then find symbol inside our active portfolio
        if isClose:
            # abs() because for orders, positions are always positive quantities
            # even though they are reported as negative shares/contracts in the
            # order status table.
            isShort = portItem.position < 0
            qty = abs(portItem.position) if (qty is None or qty == -1) else qty

            if contract is None:
                logger.error("Symbol [{}] not found in portfolio for closing!", sym)

            if isButterflyClose:
                strikesDict = await self.state.dispatch.runop(
                    "chains", sym, self.state.opstate
                )

                # strikes are in a dict by expiration date,
                # so symbol AAPL211112C00150000 will have expiration
                # 211112 with key 20211112 in the strikesDict return
                # value from the "chains" operation.
                # Note: not year 2100 compliant.
                strikes = strikes["20" + sym[-15 : -15 + 6]]

                currentStrike = float(sym[-8:]) / 1000
                pos = bisect.bisect_left(strikes, currentStrike)
                # TODO: filter this better if we are at top of chain
                (l2, l1, current, h1, h2) = strikes[pos - 2 : pos + 3]

                # verify we found the correct midpoint or else the next strike
                # calculations will be all bad
                assert (
                    current == currentStrike
                ), f"Didn't find strike in chain? {current} != {currentStrike}"

                underlying = sym[:-15]
                contractDateSide = sym[-15 : -15 + 7]

                # closing a short with butterfly is a middle BUY
                # closing a long with butterfly is a middle SELL
                if isShort:
                    # we are short a high strike, so close on lower strikes.
                    # e.g. SHORT -1 $150p, so BUY 2 $145p, SELL 1 140p
                    ratio2 = f"{underlying}{contractDateSide}{int(l1 * 1000):08}"
                    ratio1 = f"{underlying}{contractDateSide}{int(l2 * 1000):08}"
                    bOrder = f"buy 2 {ratio2} sell 1 {ratio1}"
                else:
                    # we are long a low strike, so close on higher strikes.
                    # e.g. LONG +1 $150c, so SELL 2 $155c, BUY 1 $160c
                    ratio2 = f"{underlying}{contractDateSide}{int(h1 * 1000):08}"
                    ratio1 = f"{underlying}{contractDateSide}{int(h2 * 1000):08}"
                    bOrder = f"sell 2 {ratio2} buy 1 {ratio1}"

                # use 'bOrder' for 'sym' string reporting and logging going forward
                sym = bOrder
                logger.info("[{}] Requesting butterfly order: {}", sym, bOrder)
                orderReq = self.state.ol.parse(bOrder)
                contract = await self.state.bagForSpread(orderReq)
        else:
            contract = contractForName(sym)

        if contract is None:
            logger.error("Not submitting order because contract can't be formatted!")
            return None

        if not isButterflyClose:
            # butterfly spread is already qualified when created above
            await self.state.qualify(contract)

        return await self.state.placeOrderForContract(
            sym, isLong, contract, qty, price, orderType
        )

        # TODO: allow opt-in to midpoint price adjustment following.


@dataclass
class IOpCachedQuote(IOp):
    """Return last cached value of a subscribed quote symbol."""

    def argmap(self):
        return [DArg("*symbols")]

    async def run(self):
        for sym in self.symbols:
            # if asked in "future format", drop the slash
            # TODO: should the slash removal be inside currentQuote?
            # TODO: should the slashes be part of the quote symbol name anyway?
            if sym.startswith("/"):
                sym = sym[1:]

            self.state.currentQuote(sym)


@dataclass
class IOpOrderCancel(IOp):
    """Cancel waiting orders via order ids or interactive prompt."""

    def argmap(self):
        return [
            DArg(
                "*orderids",
                lambda xs: [int(x) for x in xs],
                errmsg="Order IDs must be integers!",
            )
        ]

    async def run(self):
        # If order ids not provided via command, show a selectable list of all orders
        if not self.orderids:
            ords = [
                CB(
                    "Orders to Cancel",
                    choices=[
                        Choice(
                            f"{o.order.action} {o.contract.localSymbol} {o.order.totalQuantity} ${o.order.lmtPrice:.2f} == ${o.order.totalQuantity * o.order.lmtPrice * (100 if o.contract.secType == 'OPT' else 1):,.2f}",
                            o.order,
                        )
                        for o in sorted(self.ib.openTrades(), key=tradeOrderCmp)
                    ],
                )
            ]
            oooos = await self.state.qask(ords)

            if not oooos:
                logger.info("Cancel canceled by user cancelling")
                return

            oooo = oooos["Orders to Cancel"]
        else:
            # else, use order IDs given on command line to find existing orders to cancel
            oooo = []
            for orderid in self.orderids:
                # These aren't indexed in anyway, so just n^2 search, but the
                # count is likely to be tiny overall.
                # TODO: we could maintain a cache of active orders indexed by orderId
                for o in self.ib.orders():
                    if o.orderId == orderid:
                        oooo.append(o)

        if oooo:
            for o in oooo:
                logger.info("Canceling order {}", o)
                self.ib.cancelOrder(o)


@dataclass
class IOpBalance(IOp):
    """Return the currently cached account balance summary."""

    def argmap(self):
        return []

    async def run(self):
        ords = self.state.summary
        logger.info("{}", pp.pformat(ords))


@dataclass
class IOpPositions(IOp):
    """Print datatable of all positions."""

    def argmap(self):
        return [DArg("*symbols")]

    def totalFrame(self, df, costPrice=False):
        # Add new Total index row as column sum (axis=0 is column sum; axis=1 is row sum)
        totalCols = [
            "position",
            "marketValue",
            "totalCost",
            "unrealizedPNL",
            "dailyPNL",
            "%",
        ]

        # For spreads, we want to sum the costs/prices since they
        # are under the same order (hopefully).
        if costPrice:
            totalCols.extend(["averageCost", "marketPrice"])

        df.loc["Total"] = df[totalCols].sum(axis=0)
        t = df.loc["Total"]

        # don't do total maths unless totals actually exist
        # (e.g. if you have no open positions, this math obviously can't work)
        if t.all():
            df.loc["Total", "%"] = (
                (t.marketValue - t.totalCost) / ((t.marketValue + t.totalCost) / 2)
            ) * 100

        # Calculated weighted percentage ownership profit/loss...
        df["w%"] = df["%"] * (abs(df.totalCost) / df.loc["Total", "totalCost"])
        df.loc["Total", "w%"] = df["w%"].sum()

        if not self.symbols:
            # give actual price columns more detail for sub-penny prices
            # but give aggregate columns just two decimal precision
            detailCols = [
                "marketPrice",
                "averageCost",
                "marketValue",
                "strike",
            ]
            simpleCols = [
                "%",
                "w%",
                "unrealizedPNL",
                "dailyPNL",
                "totalCost",
            ]

            df.loc[:, detailCols] = df[detailCols].applymap(lambda x: fmtPrice(x))
            df.loc[:, simpleCols] = df[simpleCols].applymap(lambda x: f"{x:,.2f}")

            # show fractional shares only if they exist
            defaultG = ["position"]
            df.loc[:, defaultG] = df[defaultG].applymap(lambda x: f"{x:,.10g}")

        df = df.fillna("")

        # manually override the string-printed 'nan' from .applymap() of totalCols
        # for columns we don't want summations of.
        df.at["Total", "closeOrder"] = ""

        if not costPrice:
            df.at["Total", "marketPrice"] = ""
            df.at["Total", "averageCost"] = ""

        return df

    async def run(self):
        ords = self.ib.portfolio()
        # logger.info("port: {}", pp.pformat(ords))

        backQuickRef = []
        populate = []
        for o in ords:  # , key=lambda p: p.contract.localSymbol):
            backQuickRef.append((o.contract.secType, o.contract.symbol, o.contract))

            # Nice debug output showing full contracts.
            # TODO: enable global debug flags for showing these
            # maybe just enable logger.debug mode via a command?
            # logger.info("{}", o.contract)

            make = {}

            # 't' used for switching on OPT/WAR/STK/FUT types later too.
            t = o.contract.secType

            make["type"] = t
            make["sym"] = o.contract.symbol

            if self.symbols and make["sym"] not in self.symbols:
                continue

            # logger.info("contract is: {}", o.contract)
            if isinstance(o.contract, Warrant) or isinstance(o.contract, Option):
                try:
                    make["date"] = pendulum.parse(
                        o.contract.lastTradeDateOrContractMonth
                    ).date()
                except:
                    logger.error("Row didn't have a good date? {}", o)
                    pass
                make["strike"] = o.contract.strike
                make["PC"] = o.contract.right
            make["exch"] = o.contract.primaryExchange[:3]
            make["position"] = o.position
            make["marketPrice"] = o.marketPrice

            close = self.state.orderPriceForContract(o.contract, o.position)

            # if it's a list of tuples, break them by newlines for display
            if isinstance(close, list):
                closingSide = " ".join([str(x) for x in close])
            else:
                closingSide = close

            make["closeOrder"] = closingSide
            make["marketValue"] = o.marketValue
            make["totalCost"] = o.averageCost * o.position
            make["unrealizedPNL"] = o.unrealizedPNL
            try:
                make["dailyPNL"] = self.state.pnlSingle[o.contract.conId].dailyPnL
            except:
                logger.warning("No PNL for: {}", pp.pformat(o))
                # spreads don't like having daily PNL?
                pass

            if t == "FUT":
                # multiple is 5 for micros and 10 for minis
                mult = int(o.contract.multiplier)
                make["averageCost"] = o.averageCost / mult
                make["%"] = (o.marketPrice * mult - o.averageCost) / o.averageCost * 100
            elif t == "BAG":
                logger.info("available: {}", o)
            elif t == "OPT":
                # For some reason, IBKR reports marketPrice
                # as the contract price, but averageCost as
                # the total cost per contract. shrug.
                make["%"] = (o.marketPrice * 100 - o.averageCost) / o.averageCost * 100

                # show average cost per share instead of per contract
                # because the "marketPrice" live quote is the quote
                # per share, not per contract.
                make["averageCost"] = o.averageCost / 100
            else:
                make["%"] = (o.marketPrice - o.averageCost) / o.averageCost * 100
                make["averageCost"] = o.averageCost

            # if short, our profit percentage is reversed
            if o.position < 0:
                make["%"] *= -1
                make["averageCost"] *= -1
                make["marketPrice"] *= -1

            populate.append(make)
        # positions() just returns symbol names, share count, and cost basis.
        # portfolio() returns PnL details and current market prices/values
        df = pd.DataFrame(
            data=populate,
            columns=[
                "type",
                "sym",
                "PC",
                "date",
                "strike",
                "exch",
                "position",
                "averageCost",
                "marketPrice",
                "closeOrder",
                "marketValue",
                "totalCost",
                "unrealizedPNL",
                "dailyPNL",
                "%",
            ],
        )

        df.sort_values(by=["date", "sym"], ascending=True, inplace=True)

        # re-number DF according to the new sort order
        df.reset_index(drop=True, inplace=True)

        allPositions = self.totalFrame(df.copy())
        printFrame(allPositions, "All Positions")

        # attempt to find spreads by locating options with the same symbol
        symbolCounts = df.pivot_table(index=["type", "sym"], aggfunc="size")

        spreadSyms = set()
        for (postype, sym), symCount in symbolCounts.items():
            if postype == "OPT" and symCount > 1:
                spreadSyms.add(sym)

        # print individual frames for each spread since the summations
        # will look better
        for sym in spreadSyms:
            spread = df[(df.type == "OPT") & (df.sym == sym)]
            spread = self.totalFrame(spread.copy(), costPrice=True)
            printFrame(spread, f"[{sym}] Potential Spread Identified")

            matchingContracts = [
                contract
                for type, bqrsym, contract in backQuickRef
                if type == "OPT" and bqrsym == sym
            ]

            # transmit the size of the spread only if all are the same
            # TODO: figure out how to do this for butterflies, etc
            equality = 0
            if spread.loc["Total", "position"] == "0":  # yeah, it's a string here
                equality = spread.iloc[0]["position"]

            closeit = self.state.orderPriceForSpread(matchingContracts, equality)
            logger.info("Potential Closing Side: {}", closeit)


@dataclass
class IOpOrders(IOp):
    """Show all currently active orders."""

    def argmap(self):
        return []

    async def run(self):
        ords = self.ib.openTrades()

        # Note: we extract custom fields here because the default is
        #       returning Order, Contract, and OrderStatus objects
        #       which don't print nicely on their own.
        populate = []

        # Fields: https://interactivebrokers.github.io/tws-api/classIBApi_1_1Order.html
        # Note: no sort here because we sort the dataframe before showing
        for o in ords:
            # don't print canceled/rejected/inactive orders
            if o.log[-1].status == "Inactive":
                continue

            make = {}
            make["id"] = o.order.orderId
            make["sym"] = o.contract.symbol
            parseContractOptionFields(o.contract, make)
            make["occ"] = (
                o.contract.localSymbol.replace(" ", "")
                if len(o.contract.localSymbol) > 15
                else ""
            )
            make["xchange"] = o.contract.exchange
            make["action"] = o.order.action
            make["orderType"] = o.order.orderType
            make["qty"] = o.order.totalQuantity
            make["cashQty"] = (
                f"{o.order.cashQty:,.2f}" if isset(o.order.cashQty) else None
            )
            make["lmt"] = f"{o.order.lmtPrice:,.2f}"
            make["aux"] = f"{o.order.auxPrice:,.2f}"
            make["trail"] = (
                f"{o.order.trailStopPrice:,.2f}"
                if isset(o.order.trailStopPrice)
                else None
            )
            make["tif"] = o.order.tif
            # make["oca"] = o.order.ocaGroup
            # make["gat"] = o.order.goodAfterTime
            # make["gtd"] = o.order.goodTillDate
            # make["status"] = o.orderStatus.status
            make["rem"] = o.orderStatus.remaining
            make["filled"] = o.order.totalQuantity - o.orderStatus.remaining
            make["4-8"] = o.order.outsideRth
            if o.order.action == "SELL":
                if o.contract.secType == "OPT":
                    make["lreturn"] = (
                        int(o.order.totalQuantity) * float(o.order.lmtPrice) * 100
                    )
                else:
                    make["lreturn"] = int(o.order.totalQuantity) * float(
                        o.order.lmtPrice
                    )
            elif o.order.action == "BUY":
                if o.contract.secType == "OPT":
                    make["lcost"] = (
                        int(o.orderStatus.remaining) * float(o.order.lmtPrice) * 100
                    )
                elif o.contract.secType == "BAG":
                    # is spread, so we need to print more details than just one strike...
                    myLegs: list[str] = []

                    make["lcost"] = (
                        int(o.order.totalQuantity) * float(o.order.lmtPrice) * 100
                    )

                    for x in o.contract.comboLegs:
                        cachedName = self.state.conIdCache.get(x.conId)

                        # if ID -> Name not in the cache, create it
                        if not cachedName:
                            await self.state.qualify(Contract(conId=x.conId))

                        # now the name will be in the cache!
                        lsym = self.state.conIdCache[x.conId].localSymbol
                        lsymsym, lsymrest = lsym.split()
                        myLegs.append(
                            (
                                x.action[0],  # 0
                                x.ratio,  # 1
                                # Don't need symbol because row has symbol...
                                # lsymsym,
                                lsymrest[-9],  # 2
                                str(pendulum.parse("20" + lsymrest[:6]).date()),  # 3
                                round(int(lsymrest[-8:]) / 1000, 2),  # 4
                                lsym.replace(" ", ""),  # 5
                            )
                        )

                    # if all P or C, make it top level
                    if all(l[2] == myLegs[0][2] for l in myLegs):
                        make["PC"] = myLegs[0][2]

                    # if all same date, make it top level
                    if all(l[3] == myLegs[0][3] for l in myLegs):
                        make["date"] = myLegs[0][3]

                    # if all same strike, make it top level
                    if all(l[4] == myLegs[0][4] for l in myLegs):
                        make["strike"] = myLegs[0][4]

                    make["legs"] = myLegs
                else:
                    make["lcost"] = int(o.order.totalQuantity) * float(o.order.lmtPrice)

            # Convert UTC timestamps to ET / Exchange Time
            # (TradeLogEntry.time is already a python datetime object)
            make["log"] = [
                (
                    pendulum.instance(l.time).in_tz("US/Eastern"),
                    l.status,
                    l.message,
                )
                for l in o.log
            ]

            populate.append(make)
        # fmt: off
        df = pd.DataFrame(
            data=populate,
            columns=["id", "action", "sym", 'PC', 'date', 'strike',
                     "xchange", "orderType",
                     "qty", "cashQty", "filled", "rem", "lmt", "aux", "trail", "tif",
                     "4-8", "lreturn", "lcost", "occ", "legs", "log"],
        )

        # fmt: on
        if df.empty:
            logger.info("No orders!")
        else:
            df.sort_values(by=["date", "sym"], ascending=True, inplace=True)

            df = df.set_index("id")
            fmtcols = ["lreturn", "lcost"]
            df.loc["Total"] = df[fmtcols].sum(axis=0)
            df = df.fillna("")
            df.loc[:, fmtcols] = df[fmtcols].applymap(
                lambda x: f"{x:,.2f}" if isinstance(x, float) else x
            )

            toint = ["qty", "filled", "rem"]
            df[toint] = df[toint].applymap(lambda x: f"{x:,.0f}" if x else "")
            df[["4-8"]] = df[["4-8"]].applymap(lambda x: True if x else "")

            printFrame(df)


@dataclass
class IOpSound(IOp):
    """Stop or start the default order sound"""

    def argmap(self):
        return []

    async def run(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        else:
            pygame.mixer.music.play()


@dataclass
class IOpExecutions(IOp):
    """Display all executions including commissions and PnL."""

    def argmap(self):
        return []

    async def run(self):
        # "Fills" has:
        # - contract
        # - execution (price at exchange)
        # - commissionReport (commission and PNL)
        # - time (UTC)
        # .executions() is the same as just the 'execution' value in .fills()
        fills = self.ib.fills()
        # logger.info("Fills: {}", pp.pformat(fills))
        contracts = []
        executions = []
        commissions = []
        for f in fills:
            contracts.append(f.contract)
            executions.append(f.execution)
            commissions.append(f.commissionReport)

        use = []
        for name, l in [
            ("Contracts", contracts),
            ("Executions", executions),
            ("Commissions", commissions),
        ]:
            df = pd.DataFrame(l)
            if df.empty:
                logger.info("No {}", name)
            else:
                use.append((name, df))

        if use:
            df = pd.concat({name: frame for name, frame in use}, axis=1)

            # Remove all-zero and all-empty columns and all-None...
            df = df.loc[:, df.any(axis=0)]

            # Goodbye multiindex...
            df.columns = df.columns.droplevel(0)

            # Remove duplicate columns...
            df = df.loc[:, ~df.columns.duplicated()]

            # these may have been removed if no options exist,
            # or these may not exist for buy-only transactions (PNL, etc).
            for x in ["strike", "right", "date", "realizedPNL"]:
                df[x] = 0

            df["c_each"] = df.commission / df.shares

            df.loc["med"] = df[["c_each", "shares", "price", "avgPrice"]].median()
            df.loc["mean"] = df[["c_each", "shares", "price", "avgPrice"]].mean()
            df.loc["sum"] = df[
                ["shares", "price", "avgPrice", "commission", "realizedPNL"]
            ].sum()

            needsPrices = "c_each shares price avgPrice commission realizedPNL".split()
            df[needsPrices] = df[needsPrices].applymap(fmtPrice)

            # convert contract IDs to integers (and fill in any missing
            # contract ids with placeholders so they don't get turned to
            # strings with the global .fillna("") below).
            df.conId = df.conId.fillna(-1).astype(int)
            df.fillna("", inplace=True)

            df.rename(columns={"lastTradeDateOrContractMonth": "date"}, inplace=True)
            # ignoring: "execId" (long string for execution recall) and "permId" (???)

            df = df[
                (
                    """ secType conId symbol strike right date exchange localSymbol tradingClass time
             side  shares  price    orderId  cumQty  avgPrice
             lastLiquidity commission c_each realizedPNL""".split()
                )
            ]

            printFrame(df, "Execution Summary")


@dataclass
class IOpQuotesAdd(IOp):
    """Add live quotes to display."""

    def argmap(self):
        return [DArg("*symbols")]

    async def run(self):
        if not self.symbols:
            return

        ors: list[buylang.OrderRequest] = []
        for sym in self.symbols:
            # don't double subscribe
            if sym.upper() in self.state.quoteState:
                continue

            orderReq = self.state.ol.parse(sym)
            ors.append(orderReq)  # contractForName(sym))

        # technically not necessary for quotes, but we want the contract
        # to have the full '.localSymbol' designation for printing later.
        # await self.state.qualify(*cs)
        cs: list[Contract] = await asyncio.gather(
            *[self.state.contractForOrderRequest(o) for o in ors]
        )

        qs = []
        for ordReq, contract in zip(ors, cs):
            if not contract:
                logger.error("Failed to find live contract for: {}", ordReq)
                continue

            # logger.info("Adding quotes for: {} :: {}", ordReq, contract)
            tickFields = tickFieldsForContract(contract)

            # remove spaces from OCC-like symbols for key reference
            symkey = lookupKey(contract)

            self.state.quoteState[symkey] = self.ib.reqMktData(contract, tickFields)
            self.state.quoteContracts[symkey] = contract

            qs.append(symkey)

        # return array of quote lookup keys
        # (because things like spreads have weird keys we construct here the caller
        #  can then use to index into the quoteState[] dict directly later)
        return qs
        # TODO: save current quote state to global restore state


@dataclass
class IOpQuotesAddFromOrderId(IOp):
    """Add symbols for current orders to the quotes view."""

    def argmap(self):
        return [DArg("*orderIds", lambda xs: [int(x) for x in xs])]

    async def run(self):
        trades = self.ib.openTrades()

        if not self.orderIds:
            addTrades = trades
        else:
            addTrades = []
            for oid in self.orderIds:
                useTrade = None
                for t in trades:
                    if t.order.orderId == oid:
                        useTrade = t
                        break

                if not useTrade:
                    logger.error("No order found for id {}", oid)
                    continue

                addTrades.append(useTrade)

        for useTrade in addTrades:
            tickFields = tickFieldsForContract(useTrade.contract)

            # If this is a new session and we haven't previously cached the
            # contract id -> name mappings, we need to look them all up now
            # or else the next print of the quote toolbar will throw lots
            # of missing key exceptions when trying to find names.
            if useTrade.contract.comboLegs:
                for x in useTrade.contract.comboLegs:
                    # if ID -> Name not in the cache, create it
                    if not x.conId in self.state.conIdCache:
                        await self.state.qualify(Contract(conId=x.conId))
            else:
                if useTrade.contract.conId not in self.state.conIdCache:
                    await self.state.qualify(Contract(conId=useTrade.contract.conId))

            symkey = lookupKey(useTrade.contract)

            self.state.quoteState[symkey] = self.ib.reqMktData(
                useTrade.contract, tickFields
            )
            self.state.quoteContracts[symkey] = useTrade.contract


@dataclass
class IOpQuotesRemove(IOp):
    """Remove live quotes from display."""

    def argmap(self):
        return [DArg("*symbols")]

    async def run(self):
        for sym in self.symbols:
            sym = sym.upper()
            if len(sym) > 30:
                # this is a combo request, so we need to evaluate, resolve, then key it
                orderReq = self.state.ol.parse(sym)
                contract = await self.state.contractForOrderRequest(orderReq)
            else:
                # else, just a regular one-symbol lookup
                # logger.warning("QCs are: {}", pp.pformat(self.state.quoteContracts))
                contract = self.state.quoteContracts.get(sym)

            if contract:
                try:
                    self.ib.cancelMktData(contract)

                    symkey = lookupKey(contract)
                    del self.state.quoteContracts[symkey]
                    del self.state.quoteState[symkey]
                except:
                    # user requested removal of non-subscribed quote
                    # (which is still okay)
                    logger.exception("no go?")
                    pass


@dataclass
class IOpOrderSpread(IOp):
    """Place a spread order described by using BuyLang/OrderLang"""

    def argmap(self):
        return [DArg("*legdesc")]

    async def run(self):
        promptPosition = [
            Q("Symbol", value=" ".join(self.legdesc)),
            Q("Price"),
            Q("Quantity"),
            ORDER_TYPE_Q,
        ]

        got = await self.state.qask(promptPosition)

        try:
            req = got["Symbol"]
            orderReq = self.state.ol.parse(req)
            qty = int(got["Quantity"])
            price = float(got["Price"])
            orderType = got["Order Type"]
            # It appears spreads with IBKR always have "BUY" order action, then the
            # credit/debit is addressed by negative or positive prices.
            # (i.e. you can't "short" a spread and I guess closing the spread is
            #       just an "inverse BUY" in their API's view)
            order = orders.IOrder("BUY", qty, price, outsiderth=False).order(orderType)
        except:
            logger.warning("Order canceled due to incomplete fields")
            return None

        bag = await self.state.bagForSpread(orderReq)

        trade = await self.ib.whatIfOrderAsync(bag, order)
        logger.info("Impact: {}", pp.pformat(trade))

        trade = self.ib.placeOrder(bag, order)
        logger.info("Trade: {}", pp.pformat(trade))

        # self.ib.reqMarketDataType(2)
        # self.state.quoteState["THEBAG"] = self.ib.reqMktData(bag)
        # self.state.quoteContracts["THEBAG"] = bag


@dataclass
class IOpOptionChain(IOp):
    """Print option chains for symbol"""

    def argmap(self):
        return [DArg("symbol")]

    async def run(self):
        # Index cache by symbol AND current date because strikes can change
        # every day even for the same expiration if there's high volatility.
        now = pendulum.now()
        cacheKey = ("strike", self.symbol, now.date())
        # logger.info("Looking up {}", cacheKey)
        if found := self.cache.get(cacheKey):
            # logger.info("[{}] Strikes cached: {}", self.symbol, pp.pformat(found))
            return found

        contractExact = contractForName(self.symbol)

        # if we want to request ALL chains, only provide underlying
        # symbol then mock it as "OPT" so the IBKR resolver sees we
        # are requesting option underlying and not just single-stock
        # contract details.
        contractExact.secType = "OPT"

        # If full option symbol, get all strikes for the date of the symbol
        if isinstance(contractExact, (Option, FuturesOption)):
            contractExact.strike = 0.00
            # if is option already, use exact date of option...
            useDates = [pendulum.parse("20" + self.symbol[-15 : -15 + 6]).date()]
        else:
            # PERFORMANCE HACK:
            # Only request chains for two months out in an attempt to stop
            # the IBKR API from rate limiting our requests.
            # It doesn't really work though. For reliable strike and expiration
            # data not taking potentially multiple minutes to return values,
            # use external API like Tradier's market metdata APIs.
            # Also note: requesting SPY/IWM/QQQ/DIA can cause the gateway
            # to consume gigabytes of memory and lock up or crash because...
            # IBKR is shit at handling data apparently.

            # for now, don't request forward months because we are
            # still only doing short term usage (at most 1-2 weeks out).
            # Revisit end of month discovery and refactor to prefer tradier
            # API fetching first since we can get those in 100ms instead of
            # 6+ seconds for IBKR APIs sometimes.
            FORWARD_MONTHS = 0
            useDates = [
                d.date()
                for d in pendulum.period(now, now.add(months=FORWARD_MONTHS)).range(
                    "months"
                )
            ]

        # this request takes between 1 second and 60 seconds depending on ???
        # does IBKR rate limit this endpoint? sometimes the same request
        # takes 1-5 seconds, other times it takes 45-65 seconds.
        # Maybe it's rate limited to one "big" call per minute?
        # (UPDATE: yeah, the docs say:
        #  "due to the potentially high amount of data resulting from such
        #   queries this request is subject to pacing. Although a request such
        #   as the above one will be answered immediately, a similar subsequent
        #   one will be kept on hold for one minute.")
        # So this endpoint is a jerk. You may need to rely on an external data
        # provider if you want to gather full chains of multiple underlyings
        # without waiting 1 minute per symbol.
        # Depending on what you ask for with the contract, it can return between
        # one row (one date+strike), dozens of rows (all strikes for a date),
        # or thousands of rows (all strikes at all future dates).
        strikes = defaultdict(list)
        for d in useDates:
            contractExact.lastTradeDateOrContractMonth = f"{d.year}{d.month:02}"
            logger.info(
                "[{}{}] Fetching strikes...",
                self.symbol,
                contractExact.lastTradeDateOrContractMonth,
            )
            chainsExact = await self.ib.reqContractDetailsAsync(contractExact)

            # group strike results by date
            logger.info(
                "[{}{}] Populating strikes...",
                self.symbol,
                contractExact.lastTradeDateOrContractMonth,
            )

            for d in chainsExact:
                strikes[d.contract.lastTradeDateOrContractMonth].append(
                    d.contract.strike
                )

        # cleanup the results because they were received in an
        # arbitrary order, but we want them sorted for bisecting
        # and just nice viewing.
        for k, v in strikes.items():
            # also reduce to a set first to drop all the duplicate
            # call/put strikes.
            strikes[k] = sorted(set(v))

        # logger.info("Saving into {}", cacheKey)
        self.cache.set(cacheKey, strikes, expire=86400)
        logger.info("Strikes: {}", pp.pformat(strikes))

        if False:
            df = pd.DataFrame(chainsExact)
            printFrame(df)

        return strikes


@dataclass
class IOpQuoteSave(IOp):
    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        self.cache.set(cacheKey, set(self.symbols))
        logger.info("[{}] {}", self.group, self.symbols)

        repopulate = [f'"{x}"' for x in self.symbols]
        await self.state.dispatch.runop("add", " ".join(repopulate), self.state.opstate)


@dataclass
class IOpQuoteAppend(IOp):
    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)
        if not symbols:
            logger.error(
                "[{}] No quote group found. Creating new quote group!", self.group
            )
            symbols = set()

        self.cache.set(cacheKey, symbols | set(self.symbols))
        repopulate = [f'"{x}"' for x in self.symbols]
        await self.state.dispatch.runop("add", " ".join(repopulate), self.state.opstate)


@dataclass
class IOpQuoteRemove(IOp):
    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        nocache = False
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)
        if not symbols:
            nocache = True
            symbols = self.state.quoteContracts.keys()
            logger.error(
                "[{}] No quote group found so using live quote list...", self.group
            )

        goodbye = set()
        for s in self.symbols:
            for symbol in symbols:
                if fnmatch.filter([symbol], s):
                    logger.info("Dropping quote: {}", symbol)
                    goodbye.add(symbol)

        symbols -= goodbye

        # don't *CREATE* the cache key if we didn't use the cache anyway
        if not nocache:
            self.cache.set(cacheKey, symbols)

        repopulate = goodbye | {
            self.state.symbolNormalizeIndexWeeklyOptions(f'"{x}"') for x in goodbye
        }

        logger.info("Removing quotes: {}", repopulate)

        await self.state.dispatch.runop(
            "remove", " ".join(repopulate), self.state.opstate
        )


@dataclass
class IOpQuoteRestore(IOp):
    def argmap(self):
        return [DArg("group")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)
        if not symbols:
            logger.error("No quote group found for name: {}", self.group)
            return

        repopulate = [f'"{x}"' for x in symbols]
        await self.state.dispatch.runop("add", " ".join(repopulate), self.state.opstate)


@dataclass
class IOpQuoteClean(IOp):
    def argmap(self):
        return [DArg("group")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)
        if not symbols:
            logger.error("No quote group found for name: {}", self.group)
            return

        # Find any expired option symbols and remove them
        remove = []
        now = pendulum.now().in_tz("US/Eastern")

        # if after market close, use today; else use previous day since market is still open
        if (now.hour, now.minute) < (16, 15):
            now = now.subtract(days=1)

        datecompare = now.strftime("%y%m%d")
        for x in symbols:
            if len(x) > 10:
                date = x[-15 : -15 + 6]
                if date <= datecompare:
                    logger.info("Removing expired quote: {}", x)
                    remove.append(f'"{x}"')

        # TODO: fix bug where it's not translating SPX -> SPXW properly for the live removal
        await self.state.dispatch.runop(
            "qremove", "global " + " ".join(remove), self.state.opstate
        )


# TODO: potentially split these out into indepdent plugin files?
OP_MAP = {
    "Live Market Quotes": {
        "qquote": IOpQQuote,
        "quote": IOpCachedQuote,
        "depth": IOpDepth,
        "add": IOpQuotesAdd,
        "oadd": IOpQuotesAddFromOrderId,
        "remove": IOpQuotesRemove,
        "chains": IOpOptionChain,
    },
    "Order Management": {
        "limit": IOpOrderLimit,
        "buy": IOpOrder,
        "fast": IOpOrderFast,
        "spread": IOpOrderSpread,
        "modify": IOpOrderModify,
        "evict": IOpPositionEvict,
        "cancel": IOpOrderCancel,
    },
    "Portfolio": {
        "balance": IOpBalance,
        "positions": IOpPositions,
        "orders": IOpOrders,
        "executions": IOpExecutions,
    },
    "Connection": {
        "rid": IOpRID,
    },
    "Utilities": {
        "rcheck": None,
        "future": None,
        "bars": None,
        "try": None,
        "tryf": None,
        "snd": IOpSound,
        "cash": IOpCash,
        "alias": IOpAlias,
    },
    "Quote Management": {
        "qsave": IOpQuoteSave,
        "qadd": IOpQuoteAppend,
        "qremove": IOpQuoteRemove,
        "qrestore": IOpQuoteRestore,
        "qclean": IOpQuoteClean,
    },
}


# Simple copy template for new commands
@dataclass
class IOp_(IOp):
    def argmap(self):
        return [DArg()]

    async def run(self):
        ...


@dataclass
class Dispatch:
    def __post_init__(self):
        self.d = mutil.dispatch.Dispatch(OP_MAP)

    def runop(self, *args, **kwargs) -> Coroutine:
        return self.d.runop(*args, **kwargs)
