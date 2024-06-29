""" Common order types with reusable parameter configurations."""
from dataclasses import dataclass, field

from ib_async import (
    Contract,
    Order,
    OrderCondition,
    TagValue,
    # NOTE: DO NOT add LimitOrder, MarketOrder, StopLimitOrder, etc because those are
    #       non-dataclass subclasses of the Order dataclass and we can't dataclass.replace() on
    #       any other order classes except the primary Order() superclass!
)

from typing import *

import enum
from enum import auto, Enum

from loguru import logger

# Note: all functions should have params in the same order!

# Order field details:
# https://interactivebrokers.github.io/tws-api/classIBApi_1_1Order.html


@dataclass
class IOrder:
    """A wrapper class to help organize the common order logic we want to reuse.

    This looks a bit weird because we are basically duplicating most of the Order
    fields ourself and populating them before passing them back to Order, but this
    allows us to basically generate one abstract Order request then pull out more
    specific concrete Order types with conditions/algos as needed via encapsulating
    the meta-order logic inside methods generating the actual final Order object.
    Individual field detail meanings are at:
    https://interactivebrokers.github.io/tws-api/classIBApi_1_1Order.html
    """

    action: Literal["BUY", "SELL"]

    qty: float

    # basic limit price
    lmt: float = 0.00
    lmtPriceOffset: float = 0.00

    # specify amount as spend value instead of shares or contracts
    qtycash: float = 0.00

    # aux holds anything not a limit price and not a trailing percentage:
    #   - stop price for stop / stop limit / stop with protection
    #   - trailing amounts for trailing orders (instead of .trailingPercent)
    #   - touch price on MIT
    #   - offset for pegs (treated as (bid + aux) for sell and (ask - off) for buys)
    #   - trigger price for LIT (Same as touch for MIT, when the "IT" becomes marketable)
    aux: float = 0.00

    # Note: IBKR gives a warning (but not a hard error) if assigning GTC to options.
    tif: Literal["GTC", "IOC", "FOK", "OPG", "GTD", "DAY", "Minutes"] = "GTC"

    # format for these is just YYYYMMDD HH:MM:SS and assume exchange timezone i guess
    goodtildate: str = ""
    goodaftertime: str = ""

    # trailing things.
    # Note: trailpct is a floating percentage of the current price trigger
    #       instead of using a static 'aux' amount of points for the trail.
    trailstop: float = 0.00
    trailpct: float = 0.00

    # for relative orders, offset (if any) can be a percent instead of
    # a direct amount defined in 'aux'
    percentoffset: float = 0.00

    # order type is created per-returned type in a request method

    # multi-group creation
    ocagroup: Optional[str] = None

    # 1: cancel remaining orders
    # 2: reduce remaining orders (no overfill, only on market if not active)
    # 3: reduce remaining orders (potential overfill, all on market)
    ocatype: Literal[1, 2, 3] = 1
    transmit: bool = True
    parentId: Optional[int] = None

    # seconds to wait for a private execution before forwarding
    # to the public market via SMART router
    # (used for directed orders to IBKRATS or IBUSOPT)
    postToAts: int = 0

    # stock sweeps
    sweeptofill: bool = False  # stocks and warrants only

    # Trigger methods:
    # 0 - The default value.
    #       The "double bid/ask" for OTC stocks and US options.
    #       All other orders will used the "last" function.
    # 1 - "double bid/ask", stop orders triggered on two consecutive bid/ask quotes.
    # 2 - "last", stop orders are triggered based on the last price.
    # 3 double last function.
    # 4 bid/ask function.
    # 7 last or bid/ask function.
    # 8 mid-point function.
    trigger: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8] = 3
    triggerprice: float = 0.00
    outsiderth: bool = True
    override: bool = False

    # Volatility Orders
    stockrefprice: float = 0.00
    volatility: float = 0.00
    voltype: Literal[1, 2] = 2
    continuousupdate: bool = True
    refpricetype: Literal[1, 2] = 2

    # algos
    algostrategy: Literal[
        "ArrivalPx",
        "DarkIce",
        "PctVol",
        "Twap",
        "Vwap",
        "Adaptive",
        "MinImpact",
        "BalanceImpactRisk",
        "PctVolTm",
        "PctVolSz",
        "AD",
        "ClosePx",
        "ArrivalPx",
    ] = "Adaptive"
    algoparams: Optional[list[TagValue]] = None

    # preview
    whatif: bool = False

    # conditions..
    conditions: Optional[list[OrderCondition]] = None

    def order(self, orderType: str) -> Order:
        """Return a specific Order object by name."""

        # Note: these map from IBKR ALGO NAMES to our Order implementation.
        # If you want to not use IBKR ALGO NAMES everywhere, create a different mapping from
        # your nice names to these full IBKR ALGO NAMES then translate externally before reaching here.
        omap = {
            # Basic
            "LMT": self.limit,
            "MKT": self.market,
            "STP": self.stop,
            "LIT": self.limitIfTouched,
            "MIT": self.marketIfTouched,
            "REL": self.pegPrimary,
            # One Algo
            "MIDPRICE": self.midprice,
            # Common Adaptives
            # ADAPTIVE SLOW
            "LMT + ADAPTIVE + SLOW": self.adaptiveSlowLmt,
            # ADAPTIVE FAST
            "LMT + ADAPTIVE + FAST": self.adaptiveFastLmt,
            # MARKET SLOW
            "MKT + ADAPTIVE + SLOW": self.adaptiveSlowMkt,
            # MARKET FAST
            "MKT + ADAPTIVE + FAST": self.adaptiveFastMkt,
            # More Fancy
            "MKT PRT": self.marketWithProtection,
            "MTL": self.marketToLimit,
            "STOP PRT": self.stopWithProtection,
            "PEG MID": self.pegToMidpoint,
            # Multi-Leg Orders
            "REL + MKT": self.comboPrimaryPegMkt,
            "REL + LMT": self.comboPrimaryPegLmt,
            "LMT + MKT": self.comboLmtMkt,
            # Market Auction Orders
            "MOO": self.moo,
            "MOC": self.moc,
        }

        if orderMethod := omap.get(orderType):
            return orderMethod()

        # logger guard in case people are sending bad names so we can't actually generate an order...
        logger.error(
            "[{}] Order type not found! You must pass in an implemented IBKR ALGO NAME here!",
            orderType,
        )

    def adjustForCashQuantity(self, o):
        """Check if we need to use cash instead of direct quantity.

        IBKR API allows order size as cash value optionally.
        So we check if the inbound quantity is a string starting with
        a currency spec, then use cash quantity instead of share/contract
        quantity."""

        if isinstance(self.qty, str) and self.qty.startswith("$"):
            cashqty = float(self.qty[1:])
            o.totalQuantity = 0
            o.cashQty = cashqty

    def commonArgs(self, override: dict[str, Any] = None) -> dict[str, Any]:
        common = dict(
            tif=self.tif,
            goodTillDate=self.goodtildate,
            goodAfterTime=self.goodaftertime,
            outsideRth=self.outsiderth,
            whatIf=self.whatif,
        )

        if override:
            common.update(override)

        # optionally also provide a number of seconds for the order to rest in the dark pool
        # before being routed to SMART independently.
        if self.postToAts:
            common["postToAts"] = self.postToAts

        return common

    def midprice(self) -> Order:
        # Floating MIDPRICE with no caps:
        # https://interactivebrokers.github.io/tws-api/ibalgos.html#midprice
        # Also note: midprice is ONLY for RTH usage
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            lmtPrice=self.lmt,  # API docs say "optional" but API errors out unless price given. shrug.
            orderType="MIDPRICE",
            **self.commonArgs(dict(tif="DAY")),
        )

        self.adjustForCashQuantity(o)
        return o

    def marketWithProtection(self) -> Order:
        """Market-with-Protection is for futures where it executes at market, but if not fully filled, replaces remaining quantity with an auto-selected limit offset.

        (Futures only)"""
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            orderType="MKT PRT",
            **self.commonArgs(),
        )

        self.adjustForCashQuantity(o)
        return o

    def marketToLimit(self) -> Order:
        """Market-to-Limit executes at market for top-of-book quantity then submits any remainder as a limit order at the already-executed price."""
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            orderType="MTL",
            **self.commonArgs(),
        )

        self.adjustForCashQuantity(o)
        return o

    def stopWithProtection(self) -> Order:
        """Stop-with-Protection is a stop order to submit a Market-with-Protection order when (or if) the stop price is breached.

        (Futures only)"""

        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            orderType="STOP PRT",
            auxPrice=self.aux,  # the stop price for when to trigger the market-with-protection order
            **self.commonArgs(),
        )

        self.adjustForCashQuantity(o)
        return o

    def pegToMidpoint(self) -> Order:
        """Peg-to-Midpoint is a quasi-synthetic floating order type for IBKRATS or IBUSOPT.

        Orders must be directly routed to an exchange.
        The exchange is the '.exchange' attribute of the *contract* (which we don't control here).

        If you route to IBKRATS for stocks or IBUSOPT for options the midpoint can float up and down,
        but on other lit exchanges the midpoint will only float towards the execution side (up for buys, down for sells).

        https://www.interactivebrokers.com/en/index.php?f=1058
        """
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            orderType="PEG MID",
            auxPrice=0,  # no offset from midpoint, use use bid/ask midpoint directly
            lmtPrice=self.lmt,
            # these orders must be routed as HELD since they go to the IBKR dark pool and not a lit market directly.
            # (the docs don't note if IBUSOPT orders are notHeld too, but it seems like they should be?)
            # https://interactivebrokers.github.io/tws-api/ibkrats.html
            # https://www.interactivebrokers.com/en/index.php?f=4485
            # https://www.interactivebrokers.com/en/trading/orders/reroute-to-smart.php
            notHeld=True,
            **self.commonArgs(),
        )

        self.adjustForCashQuantity(o)
        return o

    def market(self) -> Order:
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            orderType="MKT",
            **self.commonArgs(),
        )

        self.adjustForCashQuantity(o)
        return o

    def marketIfTouched(self) -> Order:
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            auxPrice=self.lmt,  # the stop price for when to trigger the market-if-touched order
            orderType="MIT",
            **self.commonArgs(),
        )

        return o

    def limitIfTouched(self) -> Order:
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            # TODO: we need to fix our order placement UI to allow the "aux/stop" vs "limit price" additions...
            lmtPrice=self.lmt,  # the limit price to submit when the stop is triggered
            auxPrice=self.lmt,  # the stop price for when to trigger the market-if-touched order
            orderType="LIT",
            **self.commonArgs(),
        )

        return o

    def adaptiveFastLmt(self) -> Order:
        # Note: adaptive can't be GTC!
        # Also means this doesn't work PM/AH
        o = Order(
            orderType="LMT",
            action=self.action,
            totalQuantity=self.qty,
            lmtPrice=self.lmt,
            algoStrategy="Adaptive",
            algoParams=[TagValue("adaptivePriority", "Urgent")],
            **self.commonArgs(dict(tif="DAY")),
        )

        self.adjustForCashQuantity(o)
        return o

    def adaptiveSlowLmt(self) -> Order:
        # Note: adaptive can't be GTC!
        # Also means this doesn't work PM/AH
        o = Order(
            orderType="LMT",
            action=self.action,
            totalQuantity=self.qty,
            lmtPrice=self.lmt,
            algoStrategy="Adaptive",
            algoParams=[TagValue("adaptivePriority", "Patient")],
            **self.commonArgs(dict(tif="DAY")),
        )

        self.adjustForCashQuantity(o)
        return o

    def adaptiveFastMkt(self) -> Order:
        # Note: adaptive can't be GTC!
        o = Order(
            orderType="MKT",
            action=self.action,
            totalQuantity=self.qty,
            algoStrategy="Adaptive",
            algoParams=[TagValue("adaptivePriority", "Urgent")],
            **self.commonArgs(dict(tif="DAY")),
        )

        self.adjustForCashQuantity(o)
        return o

    def adaptiveSlowMkt(self) -> Order:
        # Note: adaptive can't be GTC!
        o = Order(
            orderType="MKT",
            action=self.action,
            totalQuantity=self.qty,
            algoStrategy="Adaptive",
            algoParams=[TagValue("adaptivePriority", "Patient")],
            **self.commonArgs(dict(tif="DAY")),
        )

        self.adjustForCashQuantity(o)
        return o

    def stop(self) -> Order:
        o = Order(
            action=self.action,
            orderType="STP",
            totalQuantity=self.qty,
            auxPrice=self.lmt,  # Stop trigger price... (currently we are using the 'lmt' field even though this is an 'aux' API field...)
            **self.commonArgs(),
        )

        return o

    def stopLimit(self) -> Order:
        o = Order(
            orderType="STP LMT",
            action=self.action,
            totalQuantity=self.qty,
            lmtPrice=self.lmt,  # Limit price under the stop...
            auxPrice=self.aux,  # Stop trigger price...
            **self.commonArgs(),
        )

        self.adjustForCashQuantity(o)
        return o

    def trailingStopLimit(self) -> Order:
        if self.aux and self.trailpct:
            raise Exception("Can't specify both Aux and Trailing Percent!")

        # Exclusive, can't have both:
        #    auxPrice=self.aux, # TRAILING AMOUNT IN DOLLARS
        #    trailingPercent=self.trailingPercent # TRAILING AMOUNT IN PERCENT
        if self.aux:
            whichTrail = dict(auxPrice=self.aux)
        else:
            whichTrail = dict(trailingPercent=self.trailpct)

        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            lmtPriceOffset=self.lmtPriceOffset,  # HOW FAR DOWN TO START THE LIMIT ± AGAINST CURRENT PRICE (- sell, + buy)
            trailStopPrice=self.trailstop,  # IF NO UP MOVEMENT, WHEN TO TRIGGER ORDER <-- THIS IS WHAT "TRAILS"
            orderType="TRAIL LIMIT",
            **whichTrail,  # type: ignore
            **self.commonArgs(),  # type: ignore
        )

        self.adjustForCashQuantity(o)
        return o

    def limit(self) -> Order:
        o = Order(
            orderType="LMT",
            action=self.action,
            totalQuantity=self.qty,
            lmtPrice=self.lmt,
            **self.commonArgs(),
        )

        self.adjustForCashQuantity(o)
        return o

    def pegPrimary(self) -> Order:
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            lmtPrice=self.lmt,  # API docs say "optional" but API errors out unless price given. shrug.
            auxPrice=self.aux,  # aggressive opposite offset of peg, can be 0 to float freely
            orderType="REL",
            **self.commonArgs(),
        )

        self.adjustForCashQuantity(o)
        return o

    def comboPrimaryPegMkt(self) -> Order:
        """Submitted as REL, but when one leg fills, other leg is eaten at market."""
        # Explained at:
        # https://www.interactivebrokers.com/en/software/tws/usersguidebook/ordertypes/relative___market.htm
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            triggerPrice=self.triggerprice,
            orderType="REL + MKT",
            **self.commonArgs(),
        )

        self.adjustForCashQuantity(o)
        return o

    def comboPrimaryPegLmt(self) -> Order:
        """Submitted as REL, but when REL triggers, other is limit? So can't be guaranteed? This isn't described anywhere."""
        # Explained at:
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            triggerPrice=self.aux,
            orderType="REL + LMT",
            **self.commonArgs(),
        )

        self.adjustForCashQuantity(o)
        return o

    def comboLmtMkt(self) -> Order:
        """Submitted as LMT, but other leg goes market when limit hits."""
        # https://www.interactivebrokers.com/en/software/tws/usersguidebook/ordertypes/limit___market.htm
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            triggerPrice=self.lmt,
            orderType="LMT + MKT",
            **self.commonArgs(),
        )

        self.adjustForCashQuantity(o)
        return o

    def moo(self) -> Order:
        """Market-on-Open Order.

        Only specify quantity since price is determined by opening auction."""
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            orderType="MKT",
            **self.commonArgs(),
        )

        o.tif = "OPG"  # opening
        o.outsideRth = False
        return o

    def moc(self) -> Order:
        """Market-on-Close Order.

        Only specify quantity since price is determined by closing auction."""
        o = Order(
            action=self.action,
            totalQuantity=self.qty,
            orderType="MOC",
            **self.commonArgs(),
        )

        # no TIF allowed for closing auction orders, it just hits the next one
        o.tif = ""
        o.outsideRth = False
        return o


@enum.unique
class CLIOrderType(Enum):
    """IBKR Order Types

    Extracted from the IBKR Java client enum OrderType from:
    IBJts/source/JavaClient/com/ib/client/OrderType.java"""

    MKT = "MKT"
    LMT = "LMT"
    STP = "STP"
    STP_LMT = "STP LMT"
    REL = "REL"
    TRAIL = "TRAIL"
    BOX_TOP = "BOX TOP"
    FIX_PEGGED = "FIX PEGGED"
    LIT = "LIT"
    LMT_PLUS_MKT = "LMT + MKT"
    LOC = "LOC"
    MIT = "MIT"
    MKT_PRT = "MKT PRT"
    MOC = "MOC"
    MTL = "MTL"
    PASSV_REL = "PASSV REL"
    PEG_BENCH = "PEG BENCH"
    PEG_MID = "PEG MID"
    PEG_MKT = "PEG MKT"
    PEG_PRIM = "PEG PRIM"
    PEG_STK = "PEG STK"
    REL_PLUS_LMT = "REL + LMT"
    REL_PLUS_MKT = "REL + MKT"
    SNAP_MID = "SNAP MID"
    SNAP_MKT = "SNAP MKT"
    SNAP_PRIM = "SNAP PRIM"
    STP_PRT = "STP PRT"
    TRAIL_LIMIT = "TRAIL LIMIT"
    TRAIL_LIT = "TRAIL LIT"
    TRAIL_LMT_PLUS_MKT = "TRAIL LMT + MKT"
    TRAIL_MIT = "TRAIL MIT"
    TRAIL_REL_PLUS_MKT = "TRAIL REL + MKT"
    VOL = "VOL"
    VWAP = "VWAP"
    QUOTE = "QUOTE"
    PEG_PRIM_VOL = "PPV"
    PEG_MID_VOL = "PDV"
    PEG_MKT_VOL = "PMV"
    PEG_SRF_VOL = "PSV"

    # and new order types for IBKRATS only
    PEGMID = "PEGMID"
    PEGBEST = "PEGBEST"
