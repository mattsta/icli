"""Command: fast

Category: Order Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import asyncio

import aiohttp
import numpy as np
import prettyprinter as pp  # type: ignore
import whenever


@command(names=["fast"])
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

    symbol: str = field(init=False)
    direction: str = field(init=False)
    amount: float = field(init=False)
    gaps: int = field(init=False)
    atm: int = field(init=False)
    expirationAway: int = field(init=False)
    percentageRun: float = field(init=False)
    preview: list[str] = field(init=False)

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
                "atm",
                convert=int,
                desc="Number of strikes away from ATM to target for starting. Can be negative to start ITM.",
            ),
            # TODO: add feature where we could just specify "OPEX" for next opex? Just need to calculate the next 3rd fridayd date.
            DArg(
                "expirationAway",
                convert=int,
                desc="use N next expiration date (0 is NEAREST, 1 is NEXT, 2 is NEXT NEXT, ...)",
            ),
            DArg(
                "percentageRun",
                convert=float,
                # don't go negative or we infinite loop!
                # TODO: allow INNER (negative) percentages/offsets so we back up with ITM strikes instead of only OTM strikes...
                verify=lambda x: x >= 0,
                desc="percentage from current price for strikes to buy (0.03 => buy up to 3% OTM strikes, etc) — OR — if 1+ then SKIP N STRIKES FROM ATM for first buy",
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
            initSym = self.symbol

            # see if we're using an alias and, if so, resolve it and attach it locally
            if initSym.startswith(":"):
                found, contract = self.state.quoteResolve(self.symbol)
                if not found:
                    logger.error("[{}] Symbol not found for mapping?", initSym)
                    return None

                assert found
                self.symbol = found

            # async run all URL fetches and data updates at once
            strikes = await self.runoplive(
                "chains",
                self.symbol,
            )

            if not strikes:
                logger.error("[{}] No strikes found?", self.symbol)
                return None

            initSym = self.symbol
            if initSym.upper() == "SPXW":
                initSym = "SPX"

            strikes = strikes[initSym]

            # also make sure quote for the underlying is populated...
            await self.runoplive(
                "add",
                f'"{initSym}"',
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
                strikes = strikes[self.symbol]

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
        now = whenever.ZonedDateTime.now("US/Eastern")

        # if after market close, use next day
        if (now.hour, now.minute) >= (16, 15):
            now = now.add(days=1, disambiguate="compatible")

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
        if not sstrikes:
            logger.error(
                "[{}] No strikes found? Does this symbol have options?", self.symbol
            )
            return None

        useExp = sstrikes[useExpIdx : useExpIdx + 1 + self.expirationAway][-1]

        assert useExp in strikes

        useChain = strikes[useExp]

        logger.info(
            "[{} :: {} :: {}] Using expiration {} having chain: {}",
            initSym,
            self.direction,
            useExpIdx,
            useExp,
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

            if any(np.isnan(np.array([currentLow, currentPrice]))):
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

        if currentPrice != currentPrice:
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
        firstStrikeIdx = find_nearest(useChain, currentPrice)

        buyStrikes = []

        # if puts, walk towards lower strikes instead of higher strikes.
        directionMul = 1 if usingCalls else -1

        # move strike by 'atm' positions in the direction of the request (LOWER PRICE for calls, HIGHER PRICE for puts)
        firstStrikeIdx += directionMul * self.atm

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

        # user-defined percentage gap for range buys...

        if self.percentageRun == 0:
            # If NO WIDTH specified, use current ATM pick (potentially moved by 'gaps' requested to start)
            poffset = self.gaps
            picked = useChain[firstStrikeIdx]
            logger.info("No width requested, so using: {}", picked)
            buyStrikes.append(picked)
        elif self.percentageRun >= 1:
            poffset = int(self.percentageRun)
            try:
                picked = useChain[
                    firstStrikeIdx + (poffset * (1 if usingCalls else -1))
                ]
            except:
                # lazy catch-all in case 'poffset' overflows the array extent
                picked = useChain[firstStrikeIdx]

            logger.info("Requested ATM+{} width, so using: {}", poffset, picked)
            buyStrikes.append(picked)
        else:
            pr = 1 + self.percentageRun
            while len(buyStrikes) < 1:
                # if we didn't find a boundary price, then extend it a bit more.
                # NOTE: This is still a hack because our ATR server isn't live again so we're guessing
                #       a 0.5% range?
                # TODO: allow fast buy with delta calculation (0.5 -> 0.25 etc)
                boundaryPrice = boundaryPrice * pr if usingCalls else boundaryPrice / pr

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
                        "Checking strike vs. boundary: ${:,.3f} v ${:,.3f}",
                        strike,
                        boundaryPrice,
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

        STRIKES_COUNT = len(buyStrikes)

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

        buyQty: dict[str, int] = defaultdict(int)

        remaining = self.amount
        spend = 0
        skip = 0
        while remaining > 0 and skip < STRIKES_COUNT:
            logger.info(
                "[{} :: {}] Remaining: ${:,.2f} plan {}",
                self.symbol,
                self.direction,
                remaining,
                " ".join(f"{x}={y}" for x, y in buyQty.items()) or "[none yet]",
            )
            for idx, (strike, occ) in enumerate(
                zip(
                    buyStrikes,
                    occs,
                )
            ):
                # TODO: for VIX, expire date is N, but contract date is N+1, so we need
                #       to do calendar math for "add 1 day" to VIX symbols...

                # (was this fixed by adding dual symbols to the Option constructor?)
                occForQuote = occ

                # multipler is a string of a number because of course it is.
                try:
                    qs = self.state.quoteState[occForQuote]
                except:
                    logger.error(
                        "[{}] Quote doesn't exist? Can't continue!", occForQuote
                    )
                    return None

                multiplier = float(qs.contract.multiplier or 1)

                ask = self.state.quoteState[occForQuote].ask * multiplier

                logger.info("Iterating [cost ${:,.2f}]: {}", ask, occForQuote)

                # if quote not populated, wait for it...
                try:
                    for i in range(0, 25):
                        if ask is not None:
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
            # TODO: FIX HACK NAME CRAP
            qs = self.state.quoteState[occ.replace("SPX", "SPXW")]
            limit = round((qs.bid + qs.ask) / 2, 2)

            # if for some reason bid/ask aren't populated, wait for them...
            while np.isnan(limit):
                await asyncio.sleep(0.075)
                limit = round((qs.bid + qs.ask) / 2, 2)

            # buy order algo hackery...
            algo = "AF"
            if "SPXW" in occ:
                algo = "PRTMKT"

            placement.append((occ, qty, limit, algo))
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
                self.runoplive("buy", f"{occ} {qty} {algo}")
                for occ, qty, _limit, algo in placement
            ]
        )

        logger.info("Placed results: {}", pp.pformat(placed))

        return True
