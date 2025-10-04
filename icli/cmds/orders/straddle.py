"""Command: straddle

Category: Order Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg
from mutil.numeric import roundnear

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import asyncio
import math

import prettyprinter as pp  # type: ignore
import whenever


@command(names=["straddle"])
@dataclass
class IOpStraddleQuote(IOp):
    """Generate a long straddle/strangle quote by providing symbol and width in strikes from ATM for put/call strikes (width 0 is just an ATM(ish) straddle).

    Note: ONLY adds quotes. DOES NOT place orders. You need to run your own orders from the quotes if necessary.


    Command supports multiple operating modes via custom parsing of the "widths" list:

    - straddle AAPL 0 — generate put/call straddle quote ATM
    - straddle AAPL 0 -10 — generate an iron condor with long put/call ATM then short put/call 10 points further out on each side.
    - straddle AAPL vertical call 0 10 — generate only a vertical call spread buying ATM+0 and selling +10 ATM
    - straddle AAPL vertical put -5 -10 - generate only a vertical put spread buying ATM-5 and selling ATM-10 ATM

    Shorthand can also be used:

    - straddle AAPL v c 2 4
    - straddle AAPL v p -3 -6

    You can also combine verticals because the inputs are consumed by just alternating reads:

    - strad /ES v c 10 20 v p -10 -20 — generates a combined call spread ATM+10 10 wide and a put spread ATM-10 10 wide.

    Note:
      - all offset math is calculated FROM THE CURRENT ATM STRIKE. So a 10 point wide spread is "0 10" or "10 20"
      - The "regular" straddle command takes sign into account, so spreads require alternating positve (long) and negative (short)
      - The "vertical" mode doesn't take sign into account for long/short since it knows the closer strike is long and the further strike is short.
    """

    symbol: str = ""
    widths: list[str] = field(default_factory=list)
    tradingClass: str = field(init=False)

    # Optionally allow using this command externally with a non-ATM automatic price starting position.
    overrideATM: float = 0.0

    def argmap(self):
        return [
            DArg(
                "symbol",
                convert=lambda x: x.upper(),
                verify=lambda x: (" " not in x) and not x.isnumeric(),
                desc="Underlying for contracts to use for ATM quote and leg choices (we auto-convert SPX->SPXW and NDX->NDXP)",
            ),
            # TODO: convert this to another sub-parser. for now we're doing hacks again.
            DArg(
                "*widths",
                default=["0"],
                desc="How many points apart to place your legs (can be 0 to just to strangle P/C ATM). Multiple widths can be defined. Positive widths are long and negative widths are short (all widths are priced in points away from ATM).",
            ),
        ]

    async def run(self):
        # Step 1: resolve input symbol to a contract
        # Step 2: fetch chains for underlying
        # Step 3: fetch current price for underlying
        # Step 4: Determine strikes to use for straddle
        # Step 5: build spread
        # Step 6: add quote for open spread
        # Step 6: add quote for close spread
        # Step 7: place spread or preview calculations

        # note: we 'convert' here because we call this class externally without the DArg automation
        if not self.widths:
            self.widths = ["0"]

        self.widths = list(map(str.upper, self.widths))

        # TODO: also replace this with the command handler, and update common handler to handle I: additions?
        if self.symbol[0] == ":":
            found, contract = self.state.quoteResolve(self.symbol)
            assert found

            self.symbol = found
            tradingClass = ""
        else:
            # Hacky way to provide trading class as symbol input...
            if "-" in self.symbol:
                self.symbol, tradingClass = self.symbol.split("-")
            else:
                tradingClass = ""

            # logger.info("[{}] Looking up underlying contract...", self.symbol)

            # we need to hack the symbol lookup if these are index options...
            prefix = "I:" if self.symbol in {"SPX", "NQ", "RTY", "VX", "VIX"} else ""
            contract = contractForName(f"{prefix}{self.symbol}")

            logger.info("[{}{}] Looking up: {}", prefix, self.symbol, contract)
            (contract,) = await self.state.qualify(contract)

            # Remove localSymbol because it conflicts with date placement on futures and the discovery
            # system prefers to use symbol and expiration date anyway.
            contract.localSymbol = ""

            logger.info("[{}] Found contract: {}", self.symbol, contract)
            assert contract.conId, f"Why isn't contract qualified? {contract=}"

        if not isinstance(contract, (Stock, Future, Index)):
            logger.error(
                "Spreads are only supported on stocks and futures and index options, but you tried to run a spread on: {}",
                contract,
            )
            return

        # TODO: cleanup this confusion between tradeSymbol and quotesym. Are they practically the same thing?
        tradeSymbol = contract.symbol

        # also hack here for index contracts having 'I:' prefix but we don't want it for the quote lookup.
        try:
            quotesym = lookupKey(contract)
        except:
            logger.exception("failed here?")
            # if lookup fails, we need to add the quote then try again
            await self.runoplive("add", tradeSymbol)

        # Convert different underlying symbols than trade symbols.
        if tradeSymbol == "SPX":
            tradeSymbol = "SPXW"
        elif tradeSymbol == "NDX":
            tradeSymbol = "NDXP"
        elif tradeSymbol == "VIX":
            tradeSymbol = "VIXW"
        elif isinstance(contract, Future):
            tradeSymbol = "/" + tradeSymbol

        try:
            for i in range(0, 100):
                if quote := self.state.quoteState.get(quotesym):
                    break

                logger.info(
                    "[{} :: {}] Quote doesn't exist yet, adding quote...",
                    tradeSymbol,
                    quotesym,
                )
                await self.runoplive("add", tradeSymbol)
                await asyncio.sleep(0)
        except:
            logger.exception("Failed to find symbol for adding quotes!")
            return False

        if not quote:
            logger.error("Quote didn't work? Can't do this without a quote.")
            return False

        # don't buy Stock(ES) instead of Future(ES)
        if isinstance(contract, Future):
            quotesym = "/" + contract.symbol

        strikes = await self.runoplive("chains", quotesym)
        strikes = strikes[quotesym]
        # logger.info("[{}] Got strikes: {}", quotesym, strikes)

        if not strikes:
            logger.error("[{} :: {}] No strikes found?", self.symbol, quotesym)
            return None

        # if we have a magic value requesting a "fake" atm for spread calculation, use it directly...
        if self.overrideATM:
            currentPrice = self.overrideATM
        else:
            # TODO: if after hours and quoting SPX, use SPX, use more dynamic ES offset.
            if quote.bid is None or quote.ask is None:
                if quote.last is None:
                    logger.error(
                        "No prices are populated, so we can't find a market price!"
                    )
                    return None

                # don't complain about an index not having bid/ask quotes
                if not isinstance(contract, Index):
                    logger.warning(
                        "[{}] Quotes aren't populated, so using LAST price",
                        contract.symbol,
                    )
                currentPrice = quote.last
            else:
                currentPrice = (quote.bid + quote.ask) / 2

        # for futures, round to nearest 10 mark because the liquidity for futures options
        # on the 5 increments is worse than the round 10 increments.
        # TODO: when we replace this with a full grammar, make round10 vs nearest-exact
        #       an in-line config option.
        if (
            (roundto := self.state.localvars.get("roundto"))
            and isinstance(contract, Future)
            and currentPrice > 1_000
        ):
            currentPrice = float(roundnear(int(roundto), currentPrice))

        logger.info("[{}] Using ATM price: {:,.2f}", self.symbol, currentPrice)

        logger.info(
            "Expiration dates ({}): {}",
            len(strikes),
            ", ".join(sorted(strikes.keys())),
        )
        sstrikes = sorted(strikes.keys())

        now = whenever.ZonedDateTime.now("US/Eastern")

        # if after market close, use next day
        if (now.hour, now.minute) >= (16, 0):
            now = now.add(days=1, disambiguate="compatible")

        # Note: DTE requests are in CALENDAR DAYS and not MARKET DAYS.
        # e.g. setting 5 DTE on a Friday for daily expirations gives you a
        #      Wednesday expiration (1: Sat, 2: Sun, 3: Mon, 4: Tues, 5: Wed)
        #      instead of 5 market days (which would be the next Friday).
        dte = int(self.state.localvars.get("dte", 0))
        now = now.add(whenever.days(dte), disambiguate="compatible")

        # Note: IBKR Expiration formats in the dict are *full* YYYYMMDD, not OCC YYMMDD.
        expTry = now.date()

        expirationFmt = f"{expTry.year}{expTry.month:02}{expTry.day:02}"
        useExpIdx = bisect.bisect_left(sstrikes, expirationFmt)

        # TODO: add param for expiration days away or keep fixed?
        self.expirationAway = 0
        useExp = sstrikes[useExpIdx : useExpIdx + 1 + self.expirationAway][-1]
        logger.info(
            "Using start date {} to discover nearest expiration ({} DTE): {}",
            expTry,
            dte,
            useExp,
        )

        assert useExp in strikes
        useChain = strikes[useExp]

        legs = []

        currentStrikeIdx = find_nearest(useChain, currentPrice)
        atm = useChain[currentStrikeIdx]

        type Strategy = Literal["STRADDLE", "STRANGLE", "VERTICAL", "VERT", "S", "V"]
        type PutCall = Literal["PUT", "CALL", "P", "C", "PS", "CS"]

        currentStrategy: Strategy = "STRADDLE"

        currentSide: PutCall | None = None
        currentlyLong: bool = False

        prewidth: Strategy | PutCall | str
        for prewidth in self.widths:
            # if the width designator is a mode switch, switch modes then continue
            if prewidth in {"STRADDLE", "STRANGLE", "VERTICAL", "VERT", "S", "V"}:
                currentStrategy = prewidth  # type: ignore
                currentSide = None
                continue

            match currentStrategy:
                case "STRADDLE" | "STRANGLE" | "S":
                    assert not prewidth.isalpha()
                    width = float(prewidth)

                    # reset the vert strategy state each time here in case we switch back over
                    currentSide = None
                    currentlyLong = False

                    # Get the SIGN of the input width.
                    # This is a bit weird because we _want_ to allow '-0' for short straddle/strangles, but
                    # there's no other way to detect -0 without stealing the sign of the input and placing it
                    # on another number to compare against.
                    # Copy sign of 'self.width' onto 1 then see if 1 is positive.
                    widthIsPositive = math.copysign(1, width) > 0

                    # allow us to build short spreads too to potentially combine with long spreads for iron condors
                    # if 'buildOpposite' is true (negative width) then we generate sells instead of buys
                    buildLong = widthIsPositive
                    width = abs(width)

                    # nearest ATM strike to the current quote
                    currentStrikeIdxLower = find_nearest(useChain, currentPrice - width)
                    currentStrikeIdxHigher = find_nearest(
                        useChain, currentPrice + width
                    )

                    put = useChain[currentStrikeIdxLower]
                    call = useChain[currentStrikeIdxHigher]
                    assert put <= atm <= call

                    side = "BUY" if buildLong else "SELL"
                    legs.append((side, "P", put))
                    legs.append((side, "C", call))
                case "VERTICAL" | "VERT" | "V":
                    # alternate adding long and short sides vertically. We also need to consume the put/call indicator.
                    if not currentSide:
                        assert prewidth in {"PUT", "CALL", "P", "C", "PS", "CS"}
                        currentSide = prewidth  # type: ignore
                        continue

                    assert not prewidth.isalpha()
                    width = float(prewidth)

                    currentStrikeIdx = find_nearest(useChain, currentPrice + width)
                    strike = useChain[currentStrikeIdx]
                    legs.append(
                        ("BUY" if not currentlyLong else "SELL", currentSide[0], strike)
                    )

                    # if this is a "safe put/call spread" add the extra protection leg on the SELL entry...
                    # (adding on SELL entry so this flips to a BUY when the spread is shorted)
                    if currentlyLong and currentSide in {"PS", "CS"}:
                        # wider protection for puts than calls because of skew
                        currentStrikeIdx = find_nearest(
                            useChain,
                            currentPrice
                            + (width * (2 if currentSide[0] == "P" else 1.5)),
                        )
                        strike = useChain[currentStrikeIdx]
                        legs.append(("SELL", currentSide[0], strike))

                    # flip long/short/long/short for the vertical spread buy/sell alternating
                    currentlyLong = not currentlyLong

        # only print this summary if we have a simple spread
        # TODO: abstract this out to show full metrics for larger inputs
        if currentStrategy == "STRADDLE" and len(legs) == 2:
            logger.info(
                "[{} :: {}] USING SPREAD: [put {:.2f}] <-(${:,.2f} wide)-> [atm {:.2f}] <-(${:,.2f} wide)-> [call {:.2f}]",
                self.symbol,
                tradeSymbol,
                put,
                atm - put,
                atm,
                call - atm,
                call,
            )

        # always create them in a consistent order
        legs = sorted(legs)

        logger.info("Legs Report: {}", pp.pformat(legs))
        if not legs:
            logger.error("No legs generated? What went wrong?")
            return None

        tradingClassExtension = f"-{tradingClass}" if tradingClass else ""
        spread = " ".join(
            [
                f"{buyorsell} 1 {tradeSymbol}{useExp[2:]}{direction}{int(strike * 1000):08}{tradingClassExtension}"
                for buyorsell, direction, strike in legs
            ]
        )

        # We don't need to manually parse the order request here since we're using `add` directly again.
        # orderReq = self.state.ol.parse(spread)

        # subscribe to quote for spread so we can figure out the current price for ordering...
        # Note: when adding spreads, 'add' also adds quote for each leg individually too.
        addedContracts = await self.runoplive("add", f'"{spread}"')

        return addedContracts
