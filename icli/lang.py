from dataclasses import dataclass, field
from typing import *
import enum

from ib_insync import Bag, Contract

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
import pygame

# import pprint
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

    Needs some tricks here because spreads don't have a bulit-in
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
                ", ".join([c.symbol for c in contracts]),
            )
            for contract in contracts:
                # request most up to date data available
                self.ib.reqMarketDataType(2)

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
class IOpEvict(IOp):
    """Evict a position using MIDPRICE sell order."""

    def argmap(self):
        return [
            DArg("sym"),
            DArg(
                "qty",
                int,
                lambda x: x != 0 and x >= -1,
                "qty is the exact quantity to evict (or -1 to evict entire position)",
            ),
        ]

    async def run(self):
        contract, qty, price = self.contractForPosition(
            self.sym, None if self.qty == -1 else self.qty
        )
        await self.state.qualify(contract)

        # set price floor to 3% below current live price for
        # the midprice order floor.
        limit = price / 1.03

        # TODO: fix to BUY back SHORT positions
        # (is 'qty' returned as negative from contractForPosition for short positions??)
        order = orders.IOrder("SELL", qty, limit).midprice()
        logger.info("Ordering {} via {}", contract, order)
        trade = self.ib.placeOrder(contract, order)
        logger.info("Placed: {}", pp.pformat(trade))


@dataclass
class IOpDepth(IOp):
    def argmap(self):
        return [
            DArg("sym"),
            DArg(
                "count",
                int,
                lambda x: 0 < x < 300,
                "depth checking iterations should be more than zero and less than a lot",
            ),
        ]

    async def run(self):
        (contract,) = await self.state.qualify(contractForName(self.sym))

        self.depthState = {}
        self.depthState[contract] = self.ib.reqMktDepth(contract, isSmartDepth=True)

        # now we read lists of ticker.domBids and ticker.domAsks for the depths
        # (each having .price, .size, .marketMaker)
        for i in range(0, self.count):
            t = self.depthState[contract]

            # loop for up to a second until bids or asks are populated
            for j in range(0, 100):
                if not (t.domBids or t.domAsks):
                    await asyncio.sleep(0.01)

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
                else:
                    fixedAsks = pd.DataFrame()

                fmtJoined = {"Bids": fixedBids, "Asks": fixedAsks}

                # Create an order book with high bids and low asks first.
                # Note: due to the aggregations above, the bids and asks
                #       may have different row counts. Extra rows will be
                #       marked as <NA> by pandas (and we can't fill them
                #       as blank because the cols have been coerced to
                #       specific data types via 'convert_dtypes()')
                printFrame(
                    pd.concat(fmtJoined, axis=1),
                    f"{contract.symbol} Grouped by Price",
                )

            # Note: the 't.domTicks' field is just the "update feed"
            #       which ib_insync merges into domBids/domAsks
            #       automatically, so we don't need to care about
            #       the values inside t.domTicks

            if i < self.count - 1:
                await asyncio.sleep(3)

        self.ib.cancelMktDepth(contract, isSmartDepth=True)
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
class IOpModifyOrder(IOp):
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
class IOpLimitOrder(IOp):
    def argmap(self):
        # allow symbol on command line, optionally
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
                ],
            ),
        ]

        gotside = await self.state.qask(promptSide)

        try:
            isClose = "to Close" in gotside["Side"]
            isSell = "Sell" in gotside["Side"]
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
            sym = got["Symbol"]
            qty = float(got["Quantity"])
            price = float(got["Price"])
            isLong = gotside["Side"].startswith("Buy")
            orderType = got["Order Type"]
        except:
            # logger.exception("Failed to get field?")
            logger.info("Order creation canceled by user")
            return None

        # if order is To Close, then find symbol inside our active portfolio
        if isClose:
            portitems = self.ib.portfolio()
            contract = sym.contract
            qty = sym.position if (qty is None or qty == -1) else qty

            if contract is None:
                logger.error("Symbol [{}] not found in portfolio for closing!", sym)
        else:
            contract = contractForName(sym)

        if contract is None:
            logger.error("Not submitting order because contract can't be formatted!")
            return None

        await self.state.qualify(contract)

        if not contract.conId:
            logger.error("Not submitting order because contract not qualified!")
            return None

        if isinstance(contract, Option):
            # don't break RTH with options...
            # TODO: check against extended late 4:15 ending options SPY / SPX / QQQ / etc?
            outsideRth = False
        else:
            outsideRth = True

        order = orders.IOrder(
            "BUY" if isLong else "SELL", qty, price, outsiderth=outsideRth
        ).order(orderType)

        logger.info("Ordering {} via {}", contract, order)
        trade = self.ib.placeOrder(contract, order)
        logger.info("Placed: {}", pp.pformat(trade))


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
class IOpCancelOrders(IOp):
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
            make["lmt"] = o.order.lmtPrice
            make["aux"] = o.order.auxPrice
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
                     "qty", "filled", "rem", "lmt", "aux", "trail", "tif",
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

        for contract in cs:
            if not contract:
                continue

            # HACK because we can only use delayed on VXM
            if contract.symbol == "VXM":
                # delayed
                self.ib.reqMarketDataType(3)
            else:
                # real time, but with last price if outside of hours.
                self.ib.reqMarketDataType(2)

            tickFields = tickFieldsForContract(contract)

            # remove spaces from OCC-like symbols for key reference
            symkey = lookupKey(contract)

            self.state.quoteState[symkey] = self.ib.reqMktData(contract, tickFields)
            self.state.quoteContracts[symkey] = contract


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
            self.ib.reqMarketDataType(2)
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
class IOpSpreadOrder(IOp):
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
        contractExact = contractForName(self.symbol)

        # If full option symbol, get all strikes for the date of the symbol
        if isinstance(contractExact, (Option, FuturesOption)):
            contractExact.strike = 0.00

        print(contractExact)
        chainsAll = await self.ib.reqSecDefOptParamsAsync(
            contractExact.symbol,
            contractExact.exchange,
            "FUT" if self.symbol.startswith("/") else "STK",
            contractExact.conId,
        )

        chainsExact = await self.ib.reqContractDetailsAsync(contractExact)

        # TODO: cache this result!
        strikes = sorted([d.contract.strike for d in chainsExact])

        logger.info("Strikes: {}", strikes)
        df = pd.DataFrame(chainsAll)
        printFrame(df)


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
        "limit": IOpLimitOrder,
        "spread": IOpSpreadOrder,
        "modify": IOpModifyOrder,
        "evict": IOpEvict,
        "cancel": IOpCancelOrders,
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
        "fast": None,
        "rcheck": None,
        "future": None,
        "bars": None,
        "buy": None,
        "try": None,
        "tryf": None,
        "snd": IOpSound,
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
