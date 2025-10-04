"""Command: buy

Category: Order Management
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from mutil.bgtask import BGSchedule
from mutil.dispatch import DArg
from tradeapis.orderlang import (
    Calculation,
    DecimalLongShares,
    DecimalPrice,
    DecimalShortShares,
)

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import asyncio
import re
import time

import numpy as np
import prettyprinter as pp  # type: ignore


@command(names=["buy"])
@dataclass
class IOpOrder(IOp):
    """Quick order entry with full order described on command line."""

    cmd: list[str] = field(init=False)

    # time in time.time()
    prevBidAskTime: float = 0
    prevBidAsk: tuple[float | None, float | None] = (None, None)

    async def currentBidAsk(
        self, contract, quoteKey
    ) -> tuple[float | None, float | None]:
        # if current quote is less than 500 ms old, return cached quote.
        # else, fetch new quote.
        # (sometimes we call currentBidAsk() multiple times on a new order,
        #  but we don't want to print the bid/ask output or actually fetch ig
        #  multiple times sequentially if we know it's "good enough" for now).
        if time.time() - self.prevBidAskTime < 0.50:
            return self.prevBidAsk

        try:
            for i in range(0, 35):
                bid, ask = self.state.currentQuote(quoteKey)

                # if we found something, we can continue
                if bid or ask:
                    self.prevBidAskTime = time.time()
                    self.prevBidAsk = (bid, ask)
                    return bid, ask

                logger.info(
                    "[{:>2} :: {}] Waiting for new quote to populate...", i, quoteKey
                )

                # else, wait for quote to populate because it could be new
                await asyncio.sleep(0.05)
        except:
            # quote isn't available currently, try again later
            logger.exception("No quote?")
            pass

        return None, None

    async def currentMidpointFor(
        self, contract, quoteKey, currentPrice: Decimal | None = None, segment=3
    ):
        """Calculate current midpoint but at 'segment' block.

        For 'segment' we divide the spread size into 10 equal sized blocks, and you can request which price edge of a block to return.

        This allows us to walk prices in more than just half increments (e.g. walk in 1/8th increments by 2,4,6,8 instead of only half increments).

        segment=2 is walking by 20%, 40%, 60%, 80% of the spread
        segment=3 is walking by 30%, 60%, 90% of the spread.
        segment=4 is walking by 40%, 80%
        segment=5 is the midpoint. (50% of the spread)
        segment=6 is walking by 60%
        segment=7 is walking by 70%
        segment=8 is walking by 80%

        We currently default to walking by segment 2 which is equivalent to using the lower 1/5th of the spread then moving inwards on each new re-pricing.
        """
        bid, ask = await self.currentBidAsk(contract, quoteKey)

        # if no bid, just go to the ask with no adjustments (because we have no valid range to determine)
        if bid is None and ask is not None:
            return Decimal(str(ask))

        if not (bid and ask):
            return None

        dbid = Decimal(str(bid))
        dask = Decimal(str(ask))

        # TODO: restore ability for OPENING ORDERS to reduce quantity when ordering and chasing quotes, but
        #       remember to DO NOT reduce quantity when chasing closing orders because we want to close a whole position.
        newPrice: Decimal | float | None
        if self.total.is_long:
            # if this is our FIRST midpoint request, we have no CURRENT PRICE, so we want a complete bid/ask midpoint
            # Also, if we are walking, and the price moves up faster than our walking, reset our estimate to the current market bid instead of our previous (now lower-than-bid) price.
            if not currentPrice or currentPrice < dbid:
                currentPrice = dbid

            # if this is a BUY LONG order, we want to close the gap between our initial price guess and the actual ask.
            width = dask - currentPrice
            newPrice = dbid + (width / 10 * segment)
        else:
            # set price if we weren't provided a price but ALSO re-align the price to the market if market jumped ahead of us during walking
            if not currentPrice or currentPrice > dask:
                currentPrice = dask

            # else, this is a SELL SHORT (or close long) order, so we want to start at our price and chase the bid closer.
            width = currentPrice - dbid
            newPrice = dask - (width / 10 * segment)

        # for now we just round everything up to the next increment. We could be smarter about rounding direction preference.
        assert newPrice is not None
        newPrice = await self.state.complyNear(contract, newPrice)
        assert newPrice is not None

        # automatically determine the "next price increment" for this contract using our instrument DB
        d = await self.state.tickIncrement(contract)
        # logger.info("Next price increment: {}", d)

        while newPrice == currentPrice:
            # TODO: if we reach at or below the bid or above the ask, maybe stop updating? We probably already filled and just didn't get a notice yet?
            # TODO: turn this update into its own object with: (bid, ask, midpoint, previousPriceAttempt, nextPriceAttempt)
            if self.total.is_long:
                # if LONG, chase UPSIDE
                newPrice = await self.state.complyUp(contract, newPrice + d)  # type: ignore
            else:
                # if short, chase DOWNSIDE
                # NOTE: DO NOT USE 'complyNear' on these because it DOES NOT CONVERGE on single repeated additions...
                newPrice = await self.state.complyDown(contract, newPrice - d)  # type: ignore

        return newPrice

    def argmap(self):
        return [
            DArg(
                "*cmd",
                desc="Command in Order Lang format for buying or selling or previewing operations",
            )
        ]

    async def run(self) -> FullOrderPlacementRecord | None | bool:
        """Run all the madness at once.

        Sure, this should probably be broken up into 5-7 different sub-functions, but it also
        kinda flows nicely as an entire novella structured in logical blocks one after another.
        """

        # we need one step of trickery here to add explicit in-line quotes to any
        # input params having spaces, so when we join them on spaces, we retain their original
        # "multi-element-quoted-value" instead of flattening *everything* and losing the
        # underlying positional context.
        cmd = " ".join([f"'{c}'" if " " in c else c for c in self.cmd])

        assert cmd, "Why didn't you provide a buy specification to run?"

        # parse the entire input string to this command through the requestlang/orderlang parser
        request = self.state.requestlang.parse(cmd)
        logger.info("[{}] Requesting: {}", cmd, request)

        self.symbol = request.symbol
        assert self.symbol
        assert request.qty

        contract = None
        self.symbol, contract = await self.state.positionalQuoteRepopulate(
            self.symbol, request.exchange
        )

        if not contract:
            logger.error("Contract not found for: {}", self.symbol)
            return None

        # convert orderlang spec to our (maybe legacy now) PriceOrQuantity holder (until we can refactor it better to just use OrderIntent everywhere)...
        self.total = PriceOrQuantity(
            value=float(request.qty),
            is_quantity=request.isShares,
            is_money=request.isMoney,
            is_long=request.isLong,
            exchange=request.exchange,
        )

        isLong: Final = self.total.is_long

        digits: Final = self.state.decimals(contract)

        algos: Final = self.state.idb.orderTypes(contract)

        # logger.info("using contract: {}", contract)
        if contract is None:
            logger.error("Not submitting order because contract can't be formatted!")
            return None

        @functools.cache
        def qk():
            return self.state.addQuoteFromContract(contract)

        def livePrice(currentPrice=None):
            quoteKey = qk()
            return self.currentMidpointFor(contract, quoteKey, currentPrice)

        async def resolveCalculation(c):
            """Common helper for resolving calculator string replacements, calculating, then returning."""
            logger.info("Resolving calculation request: {}", c)

            # If we have a "fancy" data request, build out our entire resolver from the if-then namespace capability.
            # Basically, these are if-then resolvers prefixed with 'data:' and suffixed with the timeframe (if applicable)
            # so instead of just 'bid' you would have 'data:bid' or instead of just 'atr 300' you would have 'data:atr:300'.
            parts = c.split(" ")
            rebuild = []

            # to allow multiple data requests, we split the input on spaces to check if any are 'data-request compliant'
            # then we re-build the results back into a final calc string for processing at the end.

            # TODO: instead of all this loop extraction parsing magic here, we _could_ alter the 'calc' syntax directly
            #       to use a special data extraction operator for a symbol like (data /NQZ4 mid) or (data /NQZ4 atr 300).
            for p in parts:
                if "data:" in p:
                    partss = re.search(r"data:([^\d\s]+)(\d+)?", p)
                    entires = re.search(r"data:([^\s]+)", p)
                    if not partss and not entires:
                        logger.error(
                            "Your data field didn't match our extraction attempt: {}", c
                        )
                        return None

                    assert partss
                    assert entires
                    parts = partss.groups()
                    entire = entires.groups()[0]

                    field = parts[0].rstrip(":")
                    timeframe = int(parts[1] or 0)

                    quoteKey = qk()
                    iticker = self.state.quoteState.get(quoteKey)
                    assert iticker, f"Ticker not subscribed for {quoteKey=}?"

                    resolver = self.state.dataExtractorForTicker(
                        iticker, field, timeframe
                    )

                    # the 'entire' string _doesn't_ include the 'data:' prefix, so we add it back here for replacing the whole data request:
                    p = p.replace("data:" + entire, str(resolver()))

                    logger.info("Replaced data:{} with value: {}", entire, p)
                elif "live" in p:
                    # shorthand for what should be equal to 'data:mid'
                    currentPrice = await livePrice()
                    p = p.replace("live", str(currentPrice))
                    logger.info(
                        "Replaced live placeholder with current midpoint: {}", p
                    )

                rebuild.append(p)

            runme = " ".join(rebuild)
            logger.info("Calculating: {}", runme)
            return self.state.calc.calc(runme)

        async def configMapper() -> Mapping[str, Decimal]:
            """Convert different config names into order-required key names.

            Also convert values (potentially calculations) for each field and comply based on contract needs.
            """

            # Trailing Stop needs three values from potentially four sources:
            #  - Trailing Percent (trailingPercent) - OR - Trailing Points (aux)
            #  - Stop Price (trailStopPrice)
            #  - Distance from adjusted Trailing Stop Price for Limit to be created (lmtPriceOffset)
            RESOLVERS = dict(
                aux="aux",
                trail="aux",
                trailingpercent="trailingPercent",
                trailpct="trailingPercent",
                trailstop="trailStopPrice",
                offset="lmtPriceOffset",
                lmtoffset="lmtPriceOffset",
                lmtoff="lmtPriceOffset",
            )

            fixed = {}

            for k, v in request.config.items():
                # first, rename the key if we used an easier helper alias for the config key name
                k = RESOLVERS.get(k, k)

                # If value is a calcluation, do any replacements then run it.
                if isinstance(v, Calculation):
                    gv = await resolveCalculation(v)
                    assert isinstance(gv, Decimal)

                    logger.info("[{}] Live calculation result: {}", k, gv)
                elif isinstance(v, (Decimal, float)):
                    assert contract
                    gv = await self.state.complyNear(contract, v)
                else:
                    gv = v

                assert isinstance(gv, Decimal)
                fixed[k] = gv

            return fixed

        def verifyAlgo(algo):
            # TODO: this needs to check algo map FOR SYMBOL not for ALL OF THEM!
            # TODO: we need a better "our local names" to "underlying IBKR allowed names" like "AF" needs to map to just "LMT" for checking...
            algoValid = algo in ALGOMAP.keys()  # and (algo in orderTypes)
            if not algoValid:
                logger.error(
                    "[{}] Requested algo not found in current algos! Tried:\n{}\nAvailable total algos: {}\nAvailable order types for this instrument: {}",
                    self.symbol,
                    pp.pformat(request),
                    pp.pformat(ALGOMAP),
                    pp.pformat(sorted(algos)),
                )

                return False

            return True

        # to be used with Order().dictPopulate(config)
        config = await configMapper()

        # if qty is None, it means user requested "all" quantity, so we need to look up total current position size
        # (this only works for single symbols; spreads don't have single symbols to look up)
        if request.qty is None:
            # TODO: it would be nice if we could find "total size" for spreads too, where a spread contract has
            #       contract legs, then we can look up each leg contract matched to the ratio of the legs for
            #       the total quantity (portfolio quantity of each leg / ratio == spread quantity)
            # TODO: we can do spread size management after ordermgr is validated properly.
            contracts = self.state.contractsForPosition(self.symbol, None)

            if not contracts:
                logger.error("No contracts found for: {}", self.symbol)
                return False

            if len(contracts) > 1:
                logger.error(
                    "More than one position found? We can only order one position! Found: {}",
                    contracts,
                )
                return False

            contract, qty, delayedEstimatedMarketPrice = contracts[0]

            # an "all" request is, by usage of "all," meaning our CURRENT position for REMOVAL, so
            # an "all" quantity is the opposite of our current holding quantity.
            if qty > 0:
                # if LONG, we use SHORT SHARES to close
                request.qty = DecimalShortShares(str(qty))
            else:
                # if SHORT, we use LONG SHARES to close
                request.qty = DecimalLongShares(str(qty))

        if not verifyAlgo(request.algo):
            return False

        profitAlgo = request.bracketProfitAlgo
        lossAlgo = request.bracketLossAlgo
        if profitAlgo and not verifyAlgo(request.bracketProfitAlgo):
            return False

        if lossAlgo and not verifyAlgo(request.bracketLossAlgo):
            return False

        assert request.algo
        am = ALGOMAP[request.algo]

        isPreview: Final = request.preview

        def safeBoundsForCurrentPosition(
            maxPctChange: float,
        ) -> tuple[float, float, float]:
            """Take a float percentage (0.10) and generate (lower, upper) price bounds based on the current average position cost.

            We *ALSO* want to validate we are not buying or selling adversarial low-depth "shock quotes" so we also want to:
              - BUY no higher than 10% off the bid
              - SELL no lower than 10% off the ask

            The result values are expected to be used as:
                allowCurrentPrice = lower <= currentPrice <= upper
            """
            averagePriceSoFar = self.state.averagePriceForContract(contract)

            # convert input 0.xx percentage to multiply-growth/divide-contract percentage weight
            adjustBounds = 1 + maxPctChange

            # if we have NO position or if we are in a CLOSING MODE (short going long-to-close; or long going short-to-close),
            # then don't enforce any price caps.
            # If we want to open or close a position, we want any prices we can get at the moment
            # (assuming the bid/ask spread isn't broken; maybe we want to limit upper/lower by a wide EMA multiple of the spread midpoint).
            lower = float("-inf")
            upper = float("inf")

            # if currently holding LONG and this is still accumulating (long + long),
            # we want to limit higher prices but allow wider lower prices.
            # (if this were a closing request, it would be long + short to close)
            if averagePriceSoFar > 0 and isLong:
                upper = averagePriceSoFar * adjustBounds
            elif averagePriceSoFar < 0 and not isLong:
                # else, if currently holding SHORT and still accumulating,
                # we want to limit lower prices from crashing our cost basis,
                # but allow higher prices to increase our short profit cost basis.
                # (if this were a closing request, it would be short + long to close)
                lower = averagePriceSoFar / adjustBounds

            return lower, upper, averagePriceSoFar

        def isSafeBounds(pctBounds: float, price: float | Decimal):
            lower, upper, avgPrice = safeBoundsForCurrentPosition(pctBounds)

            safe = lower <= price <= upper
            if not safe:
                logger.error(
                    "Safe bounds ({}) exceeded: {} <= {} <= {} (with average cost: {})",
                    pctBounds,
                    lower,
                    price,
                    upper,
                    avgPrice,
                )

            return safe, avgPrice

        useAutomaticPriceWalking = False

        # begin process of sending the order to IBKR
        # if we don't have an initial price, START FROM MIDPOINT THEN WALK AROUND

        async def isSafeBidAsk(
            pctBounds: float, currentPrice: float | Decimal
        ) -> tuple[bool, float | None, float | None]:
            quoteKey = qk()
            bid, ask = await self.currentBidAsk(contract, quoteKey)

            if not (bid or ask):
                return False, None, None

            if bid is None:
                safeBid = 0.0
            else:
                safeBid = bid + (bid * pctBounds)

            if ask is None:
                return False, None, None

            safeAsk = ask - (ask * pctBounds)

            # if the bid/ask are less than the percent width apart, they overlap so we know we are within the bounds
            if safeAsk <= safeBid:
                return True, safeBid, safeAsk

            return safeBid <= currentPrice <= safeAsk, safeBid, safeAsk

        async def checkPriceSafety(
            pctBounds: float, currentPrice: float | Decimal
        ) -> bool:
            # TODO: clean up this safety check:
            #       - allow config override to disable (or change bounds?)
            #       - combine with "don't buy more than 10% above bid; don't sell more than 10% below ask" (with overrides/config too)
            assert isinstance(request.limit, (float, Decimal))
            safe, avgPrice = isSafeBounds(pctBounds, request.limit)

            if not safe:
                logger.error("Safety current position bounds failed?")
                return False

            safebidask, safebid, safeask = await isSafeBidAsk(pctBounds, request.limit)
            if not safebidask:
                logger.error(
                    "Safety check for bid/ask distance from limit failed? Got: {} <= {} <= {}",
                    safebid,
                    request.limit,
                    safeask,
                )
                return False

            return True

        # inbound order request had no price provided, which is our signal to use automatic price walking
        if request.limit is None:
            useAutomaticPriceWalking = True

            # check live quote price (hopefully populated already)
            request.limit = await livePrice()

            # if no quote found (after hours, bad symbols, IBKR maintenance times) then we can't do anything.
            if request.limit is None:
                logger.error(
                    "[{}] No live quote found. Can't create auto-generated price. Stopping order.",
                    qk(),
                )

                return None

            # TODO: make these pctBounds configurable via... some configure option
            assert isinstance(request.limit, (float, Decimal))
            safe = await checkPriceSafety(0.10, request.limit)

            if not safe:
                return False
        elif isinstance(request.limit, Calculation):
            # if the limit is a CALCULATOR TO EXECUTE, first parse it for any replacement (live prices, etc)
            # then run the calculation.
            request.limit = await resolveCalculation(request.limit)
            logger.info("[LMT] Live calculation result: {}", request.limit)

            assert isinstance(request.limit, (float, Decimal))
            safe = await checkPriceSafety(0.10, request.limit)
            if not safe:
                return False

        # limit price _must_ be a decimal here because the original parser provides decimals and we generate decimals
        assert isinstance(
            request.limit, Decimal
        ), f"Expected decimal, but got {request.limit=}?"

        multiplier = self.state.multiplier(contract)
        dmul = Decimal(str(multiplier))

        if scale := request.scale:
            scaleAvg = request.scaleAvgRecord
            assert isinstance(scaleAvg.limit, (float, Decimal))
            assert isinstance(scaleAvg.qty, (float, Decimal))

            scaleTotal = float(scaleAvg.limit) * float(scaleAvg.qty) * multiplier
            # logger.info("scale avg is: {}", scaleAvg)

            logger.info(
                "[{}] Scale Details: total qty {} @ avg cost ${:,.{}f} == total cost ${:,.{}f} ({} x {:,.{}f} x {}); profit @ {:,.{}f} (+{} % :: ${:,.{}f}); loss @ {:,.{}f} (-{} % :: ${:,.{}f})",
                contract.localSymbol or contract.symbol,
                scaleAvg.qty,
                scaleAvg.limit,
                digits,
                scaleTotal,
                digits,
                scaleAvg.qty,
                scaleAvg.limit,
                digits,
                multiplier,
                scaleAvg.bracketProfitReal or nan,
                digits,
                scaleAvg.bracketProfit,
                ((scaleAvg.bracketProfitReal - scaleAvg.limit) * scaleAvg.qty * dmul)
                if scaleAvg.bracketProfitReal
                else nan,
                digits,
                scaleAvg.bracketLossReal or nan,
                digits,
                scaleAvg.bracketLoss,
                ((scaleAvg.limit - scaleAvg.bracketLossReal) * scaleAvg.qty * dmul)
                if scaleAvg.bracketLossReal
                else nan,
                digits,
            )

        async def updateBracket() -> Bracket | None:
            """Update bracket using current state.

            We make this a local closure because during walking order modification, we move the bracket prices
            by overwriting 'request.limit' then re-running this to calculate the new bracket offsets.
            """

            bracket = None
            assert contract is not None

            profitAlgo = request.bracketProfitAlgo
            lossAlgo = request.bracketLossAlgo

            if request.bracketProfit:
                # Create PROFIT exit at +$A.BC PROFIT
                # (profit is ABOVE your entry for a long, but BELOW your entry for a short.
                #  You desginate your PROFIT in +points in either case and we handle the sign flip automatically)

                # If value is a calcluation, do any replacements then run it.
                if isinstance(request.bracketProfit, Calculation):
                    request.bracketProfit = DecimalPrice(
                        await resolveCalculation(request.bracketProfit)
                    )
                    logger.info(
                        "[{}] Live calculation result: {}",
                        "bracketProfit",
                        request.bracketProfit,
                    )

                assert request.bracketProfitReal is not None

                extensionPriceProfit = await self.state.complyNear(
                    contract, request.bracketProfitReal
                )

                profitAlgo = ALGOMAP[profitAlgo]  # type: ignore

                # We don't need a check for an existing bracket define here
                # because this is the first bracket check case, so always create
                # if we reach here.
                bracket = Bracket(
                    profitLimit=extensionPriceProfit, orderProfit=profitAlgo
                )

            if request.bracketLoss:
                # Create LOSS exit at -$X.YZ LOSS
                # (even if short, the loss value here will be YOUR loss so ABOVE your entry)

                # If value is a calcluation, do any replacements then run it.
                if isinstance(request.bracketLoss, Calculation):
                    request.bracketLoss = DecimalPrice(
                        await resolveCalculation(request.bracketLoss)
                    )
                    logger.info(
                        "[{}] Live calculation result: {}",
                        "bracketLoss",
                        request.bracketLoss,
                    )

                assert isinstance(request.bracketLossReal, (float, Decimal))
                extensionPriceLoss = await self.state.complyNear(
                    contract, request.bracketLossReal
                )

                lossAlgo = ALGOMAP[lossAlgo]  # type: ignore

                # TODO: we should be able to configure an explicit stop activation (aux) vs. stop limit price
                #       if we are using stop limit orders for loss triggering...
                #       (right now we just default the limit price to the stop price which may not always work)
                #       (or we could make this more adaptive with trailing stop limit orders instead)
                if not bracket:
                    bracket = Bracket(lossLimit=extensionPriceLoss, orderLoss=lossAlgo)
                else:
                    # NOTE: we aren't currently consuming the 'aux' config value INTO the request object for bracket math.
                    # So, your stop 'aux' level is not adjusted here after your entry definition.

                    # else, add to existing bracket from profit entry above
                    bracket.lossLimit = extensionPriceLoss
                    bracket.lossStop = extensionPriceLoss
                    bracket.orderLoss = lossAlgo

            return bracket

        bracket = await updateBracket()

        # BUY CMD
        assert self.symbol
        placed = await self.state.placeOrderForContract(
            self.symbol,
            isLong,
            contract,
            qty=self.total,
            # if we are buying by AMOUNT, we don't specify a limit price since it will be calculated automatically,
            # then we read the calculated amount to run the auto-price-tracking attempts.
            # if we provide a price, it gets used as the limit price directly.
            limit=request.limit,
            orderType=am,
            preview=isPreview,
            bracket=bracket,
            config=config,
            ladder=Ladder.fromOrderIntent(request),
        )

        if isPreview:
            # Don't continue trying to update the order if this was just a preview request
            return placed

        if not placed:
            logger.error("[{}] Order can't continue!", self.symbol)
            return False

        # if this is a market order, don't run the algo loop
        if {
            "MOO",
            "MOC",
            "MKT",
            "LIT",
            "MIT",
            "SLOW",
            "PEG",
            "STP",
            "STOP",
            "MTL",
        } & set(am.split()):
            logger.warning(
                "Not running price algo because this is a passive or slower resting order..."
            )
            return False

        # Also don't run the algo loop if we have a MANUAL price requested:
        if not useAutomaticPriceWalking:
            logger.warning(
                "Not running price algo because limit price provided directly..."
            )
            return False

        # If we reach here, we are letting the following code LIVE UPDATE THE PRICE UNTIL ALL QTY IS FILLED

        # Extract fields from the return value of the order placement
        profitTrade = None
        profitOrder = None
        lossTrade = None
        lossOrder = None
        match placed:
            case FullOrderPlacementRecord(
                limit=TradeOrder(trade=trade, order=order),
                profit=profitRecord,
                loss=lossRecord,
            ):
                if profitRecord:
                    profitTrade = profitRecord.trade
                    profitOrder = profitRecord.order

                if lossRecord:
                    lossTrade = lossRecord.trade
                    lossOrder = lossRecord.order

        # register a CUSTOM status event handler into the trade status system so even if we aren't listening
        # during a live event emission, we still have a record "something updated" for us to check.
        event = asyncio.Event()

        # record our previous status in a dict so the closure can modify persistent state outside of itself
        statusState = dict(
            prevStatus=trade.orderStatus.status,
            prevRemaining=trade.orderStatus.remaining,
            prevMessage=trade.log[-1].message,
        )

        logger.info("Starting with trade status: {}", statusState)

        def statusUpdate(tr):
            # we need to declare 'statusState' as nonlocal because this is being called from very far away...
            # (and right now the event system doesn't have a way to pass state objects into callbacks)
            nonlocal statusState

            # Only update our event notification if something useful changed.
            # (e.g. don't updated on status going from Submitted -> Submitted each modification price update)
            logger.info("Got status update: {}", tr.orderStatus)

            if (
                tr.orderStatus.status != "ValidationError"
                and tr.orderStatus.status != statusState["prevStatus"]
            ):
                logger.info(
                    "[{} -> {}] Status updated!",
                    statusState["prevStatus"],
                    tr.orderStatus.status,
                )
                event.set()
            else:
                logger.info(
                    "[{} -> {}] Not triggering status update event because order hasn't made market progress",
                    tr.orderStatus.status,
                    statusState["prevStatus"],
                )

            message = tr.log[-1].message
            if message and message != statusState["prevMessage"]:
                logger.info(
                    "[{} :: {}] New message: {}",
                    trade.order.orderId,
                    trade.orderStatus.status,
                    message,
                )

            # update history with current state to compare against next update
            statusState |= dict(
                prevStatus=tr.orderStatus.status,
                prevRemaining=tr.orderStatus.remaining,
                prevMessage=message,
            )

        trade.statusEvent += statusUpdate

        started: Final = time.time()

        def howlong():
            """Return number of seconds since we started running this order update process.

            This isn't the exact time since the order was _placed_ since the trade is already
            working in the IBKR system by now, but this is close enough for our update purposes.
            """
            return time.time() - started

        # TODO: make price-walking wait interval more configurable.
        # We want to give every price update a _chance_ to work before smashing with new price updates.
        # TODO: make WAIT DURATION adaptive or a config parameter (futures wait less than options, etc)
        PRICE_UPDATE_WAIT_DURATION = 1.25
        MESSAGE_UPDATE_DURATION = 0.222

        # when running price updates, run when (now - previous update >= PRICE_UPDATE_WAIT_DURATION) checked every MESSAGE_UPDATE_DURATION
        # TODO: we could also make this an adaptable timer difference too I guess
        PREVIOUS_UPDATE_AT = 0.0

        def howlongevent() -> float:
            """Time elapsed between now and the previous trigger time"""
            if PREVIOUS_UPDATE_AT:
                return time.time() - PREVIOUS_UPDATE_AT

            # if no recorded update time yet, we don't have a valid duration to measure against
            return np.nan

        def readyForNextUpdateTrigger() -> bool:
            # the trigger is only valid if the previously set time is populated
            if PREVIOUS_UPDATE_AT:
                return howlongevent() >= PRICE_UPDATE_WAIT_DURATION

            return False

        def nextTriggerDuration() -> float:
            """Next time to run a price check in seconds based on our delay/wait/retry paraemters and the previous time attempt."""

            # if we have a previous update time, then the logic holds
            # (if _more_ time has passed than our duration, then the check is negative, and we return 0 for "CHECK NOW",
            #  otherwise if _less_ time has passed than our duration, then the time is "padded up to" our duration check interval again)
            if PREVIOUS_UPDATE_AT:
                return max(
                    0, PRICE_UPDATE_WAIT_DURATION - (time.time() - PREVIOUS_UPDATE_AT)
                )

            # if we don't have a previous update time, then our time to wait us the duration itself at most
            return PRICE_UPDATE_WAIT_DURATION

        # TODO: refactor all of this into a self-contained price tracking agent system...
        while not trade.isDone():
            ## BEGIN AUTOMATIC PRICE WALKING LOOP STAGE (waiting for status changes, running price updates until filled or stopped) ##
            # wait for the order to go from PreSubmitted -> Submitted.
            # It doesn't make sense to start adjusting prices until the order is actually interacting with live spreads.
            # OUTER LOOP IS FOR THE ENTIRE PRICE MANAGEMENT SYSTEM TO RUN UNTIL ALL FILLS ARE COMPLETE
            while not trade.isDone():
                # record if order is currently active on an exchange *before* the next update triggers.
                # If order is NOT active (i.e. NEW order) and it BECOMES active, we do not run price updates on the first "Submitted/active" notification
                # because we want to give the order a couple seconds to try and hit an offer instead of updating the price 10 ms after it goes live.
                currentlyWorking = trade.isWorking()

                # INNER LOOP IS ONLY WAITING FOR NEXT EVENT TO TRIGGER BEFORE CHECKING STATUS AND FILLS AND PRICES AGAIN
                while not trade.isDone():
                    try:
                        timeout = min(MESSAGE_UPDATE_DURATION, nextTriggerDuration())
                        logger.info(
                            "[{} :: {} :: lmt ${:,.{}f} :: {:,.2f} seconds since last event; {:,.2f} seconds total] Waiting {:,.2f} s for next update...",
                            trade.order.orderId,
                            trade.orderStatus.status,
                            trade.order.lmtPrice,
                            digits,
                            howlongevent(),
                            howlong(),
                            timeout,
                        )

                        # use the API event system to wait for a live event to happen.
                        # possible events are: statusEvent modifyEvent fillEvent commissionReportEvent filledEvent cancelEvent cancelledEvent

                        # if we get a status update, _something_ happened (either a status transition or a full fill a partial fill or a cancel)
                        await asyncio.wait_for(event.wait(), timeout=timeout)
                        event.clear()

                        logger.info(
                            "[{} :: {} :: {} of {}] Received status update in wait cycle... checking for completeness...",
                            trade.order.orderId,
                            trade.orderStatus.status,
                            trade.orderStatus.filled,
                            trade.orderStatus.total,
                        )

                        break
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        # catches CTRL-C during sleep
                        logger.warning(
                            "[{}] User canceled waiting before order checks completed! Order still live.",
                            trade.orderStatus.orderId,
                        )
                        return False
                    except TimeoutError:
                        # catch wait_for() timeout
                        # If we reached the next update trigger time due to timeouts (of not receiving status updates),
                        # then we need to break this loop and start updating some prices...
                        if readyForNextUpdateTrigger():
                            logger.info("Ready for next price update check...")
                            break

                        # else, this was just a timeout for us to print the next user status log line, so continue
                        continue
                    except:
                        logger.exception("Something else broke?")

                if trade.isDone():
                    logger.info(
                        "[{} :: {} :: {} of {}] Order is marked as complete! Not running further price updates.",
                        trade.order.orderId,
                        trade.orderStatus.status,
                        trade.orderStatus.filled,
                        trade.orderStatus.total,
                    )
                    break

                if trade.isWaiting():
                    logger.info(
                        "[{} :: {} :: {} of {}] Order is marked as not live yet! Not running price updates until order goes live.",
                        trade.order.orderId,
                        trade.orderStatus.status,
                        trade.orderStatus.filled,
                        trade.orderStatus.total,
                    )
                    continue

                # At this point, all we know is: the order is CURRENTLY LIVE and CURRENTLY WORKING at an exchange

                # defensive logic check. This should _never_ trigger, but if it does, it means you were
                # about to get stuck in a zero-millisecond infinite loop, so here you have a chance to
                # manually cancel your operations.
                TIME_CHECK = time.time()
                if abs(TIME_CHECK - PREVIOUS_UPDATE_AT) <= 0.010:
                    logger.error(
                        "We don't support updates this fast! Attempted loop after only {} - {} = {} seconds. Pausing to continue...",
                        TIME_CHECK,
                        PREVIOUS_UPDATE_AT,
                        TIME_CHECK - PREVIOUS_UPDATE_AT,
                    )
                    try:
                        await asyncio.sleep(0.333)
                    except:
                        logger.error(
                            "User stooped live order update tracking. Order is still active."
                        )
                        break

                # since we fell out of the event wait loop, we MUST mark this as our previous update attempt, so the next
                # time the logger triggers an update, the wait exception catch knows to not continue price activity
                # until at least PRICE_UPDATE_WAIT_DURATION has passed.
                PREVIOUS_UPDATE_AT = time.time()

                # if this is our FIRST notification of the status moving to an exchange, we reset the timer waiting
                # for our first live fill (we don't include the "pre-submitted" phase as part of our walk-wait timer).
                if (not currentlyWorking) and trade.isWorking():
                    # TODO: *should* we be targeting a NEW midpoint here if the price moved between when we placed
                    #       the order and when the order hit the exchange (can be 1-5 seconds sometimes?)
                    if trade.orderStatus.status != "Submitted":
                        logger.warning("BUG: Why isn't status Submitted here?")

                    logger.info(
                        "[{} :: {} :: {} of {}] Received first Submitted status update... pausing {} seconds to wait for live fills now...",
                        trade.order.orderId,
                        trade.orderStatus.status,
                        trade.orderStatus.filled,
                        trade.orderStatus.total,
                        PRICE_UPDATE_WAIT_DURATION,
                    )
                    # the next price update will happen in PRICE_UPDATE_WAIT_DURATION seconds from now
                    # (if the order doesn't fill before then)
                    continue

                # if order is COMPLETE then STOP TRYING
                if trade.orderStatus.remaining == 0:
                    logger.error(
                        "How is remaining quantity zero here if the order is still live?"
                    )
                    return True

                # At this point, we know:
                #  - the order is LIVE AND WORKING
                #  - the order has been LIVE for a minimum of PRICE_UPDATE_WAIT_DURATION seconds
                #  - the order IS NOT COMPLETE YET
                #  - so we UDPATE PRICES CLOSER TO THE NBBO

                # convert trade object order price to Decimal() so our math doesn't break due to type issues
                currentPrice = Decimal(str(trade.order.lmtPrice))

                # re-read current quote for symbol before each update in case the price is moving rapidly against us
                logger.info("Adjusting price for more aggressive fills...")

                if (newPrice := await livePrice(currentPrice)) is None:
                    logger.error(
                        "Failed to find new midpoint for price updating! Order still active."
                    )
                    return False

                if trade.isDone():
                    logger.warning(
                        "Trade completed while checking live price! Not updating anymore."
                    )
                    return False

                # when using automatic price walking, don't increase our cost basis by more than 10%
                # (this prevents things like if we have a cost basis of $3 and are walking up, but something
                #  suddenly offers $5 instead of $3.10, we don't grab $5 suddenly)
                # TODO: consume price walk limit as a config parameter from the order intent too
                safe, avgPrice = isSafeBounds(0.10, newPrice)
                if not safe:
                    # the isSafeBounds() reports its own error message
                    continue

                # TODO: inject logic here to, if this is a current position, don't move our cost basis against us by more than 5%? 10%?
                #       Basically: cap the maximum we can modify the price here.
                #       ALSO, this needs more accurate "broken spread" checks where if the spread is greater than 50% we don't trust it.
                #       ALSO, track the spread width itself and if it bounces from "normal" to weird/wide, maybe bail out or just stop for a while.

                logger.info(
                    "Price changing from {:,.{}f} to {:,.{}f} (difference {:,.{}f}) for spending {:,.{}f}",
                    currentPrice,
                    digits,
                    newPrice,
                    digits,
                    newPrice - currentPrice,
                    digits,
                    (float(newPrice) * trade.orderStatus.remaining),
                    digits,
                )

                if currentPrice == newPrice:
                    logger.error(
                        "Not submitting order update because no price change? Order still active!"
                    )
                    return False

                # generate NEW ORDER OBJECTS with ACCEPTABLE API PARAMETERS or else IBKR rejects updates
                request.limit = newPrice
                if bracket:
                    bracket = await updateBracket()
                    assert bracket

                    # UPDATE BRACKET IF NECESSARY
                    if profitTrade:
                        logger.info(
                            "Updating PROFIT exit: ${:,.{}f}",
                            bracket.profitLimit,
                            digits,
                        )
                        profitTrade.order = await self.state.safeModify(
                            profitTrade.contract,
                            profitTrade.order,
                            lmtPrice=bracket.profitLimit,
                        )

                    if lossTrade:
                        logger.info(
                            "Updating LOSS exit: ${:,.{}f}", bracket.lossLimit, digits
                        )
                        lossTrade.order = await self.state.safeModify(
                            lossTrade.contract,
                            lossTrade.order,
                            lmtPrice=bracket.lossLimit,
                            aux=bracket.lossStop,
                        )

                updatedOrder = await self.state.safeModify(
                    trade.contract, trade.order, lmtPrice=newPrice
                )

                if trade.isDone():
                    logger.warning(
                        "Trade completed while generating modified order! Not updating anymore."
                    )
                    return False

                logger.info(
                    "[{} :: {} :: {} of {}] Submitting order update to: {} Using: {}",
                    trade.order.orderId,
                    trade.orderStatus.status,
                    trade.orderStatus.filled,
                    trade.orderStatus.total,
                    newPrice,
                    pp.pformat(updatedOrder),
                )

                # submit our PRICE MODIFICATION to IBKR for updating our existing order on the exchange(s)
                # NOTE: we update brackets (if any) BEFORE updating the order price so we maintain the expected ± offsets from the original order.
                if profitTrade:
                    self.ib.placeOrder(profitTrade.contract, profitTrade.order)

                if lossTrade:
                    self.ib.placeOrder(lossTrade.contract, lossTrade.order)

                self.ib.placeOrder(trade.contract, updatedOrder)

        # for now, stop trying to automatically run "positions" because it was too delayed and confusing us a bit
        if False:
            # now just print current holdings so we have a clean view of what we just transacted
            # (but schedule it for a next run so so the event loop has a chance to update holdings first)
            async def delayShowPositions():
                await self.runoplive("positions", [])

            self.task_create(
                "show ord positions",
                delayShowPositions,
                schedule=BGSchedule(delay=0.77, runtimes=3, pause=0.33),
            )

        # "return True" signals to an algo caller the order is finalized without errors.
        return True
