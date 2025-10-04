"""instrumentdb is our unified interface for instrument price compliance.

The goal is to provide a simple interface using IBKR APIs and bootstrap databases to:
    - detect the number of usable decimal points per instrument (0 to 8)
    - given an input number, round to nearest valid decimal point for instrument
      - but this depends on: size being traded, target exchange, instrument being traded, price being traded
      - and all of those are only provided (reliabily) by IBKR metadata APIs
      - but we don't want to run metadata API lookups on EVERY trade attempt, so we need to look them
        all up once on FIRST attempt then cache them for a long time (so future access is near-instant locally)

Given we need to comply instruments with price increments, we also need to comply:
    - instrument to exchange mappings
    - exchange mappings to tick increment mappings
      - why? example: even though options have increment requirements $0.01 -> $0.05 or $0.05 -> $0.10,
                      there are two ways to obtain $0.01 increments for ALL sizes:
                        - brokers can always "Self-trade" between internal customers at any increment
                        - NASDAQ runs two "any increment for any instrument" option exchanges
                        - the remaining exchanges conform to the official CBOE rules
    - instrument to IBKR algo allowance mappings
    - cache tick increment details locally (marketRuleIds/ticks are reused between instruments)
    - which means we may as well also cache the entire instrument detail object also having:
      - instrument trading times (full and liquid)
      - minimum trade size
      - recommended trade size (minimum size to avoid extra commission penalties)

IBKR APIs send and receive price details as strings, so we can use Python's Decimal() object
for clean decimal manipulation because str(Decimal(x)) is exactly 'x' as compared to higher
precision floats sometimes losing resolution due to float representation details.
"""

import asyncio
import bisect
from collections.abc import Coroutine
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from ib_async import IB, Bag, Contract
from loguru import logger
from mutil.dualcache import DualCache
from mutil.numeric import ROUND, roundnear


@dataclass(slots=True)
class MarketRuleId:
    """IBKR uses 'MarketRuleId' to define the price increments instruments trade at _per exchange_"""

    ruleId: int

    # maintain two equal size lists representing [size, increment]
    minTick: tuple[float, ...]  # e.g. [0, 3]
    increment: tuple[Decimal, ...]  # e.g. [0.01, 0.05]

    # number of decimal places for rounding based on maximum increment precision!
    decimals: int

    # cache args locally to avoid re-use if possible.
    # We can't use cache decorators due to this being a dataclass.
    memoize: dict[Any, Any] = field(default_factory=dict)

    def incrementForSize(self, size: Decimal | float) -> Decimal:
        """Return the correct rounding increment for an order of size 'size' under this market rule"""
        # if only one size, don't search
        if len(self.minTick) == 1:
            return self.increment[0]

        # NOTE: it turns out some rules DO have negative sizes! ugh, thanks. Symbols like TICK-NYSE
        #       show up as:  [PriceIncrement(lowEdge=-9999.0, increment=1.0), PriceIncrement(lowEdge=0.0, increment=1.0)]
        # assert (size > 0), f"Why are you trying to increment a non-positive quantity? Asked based on size: {size}"

        # else, figure out which size we need to check...
        pos = bisect.bisect_left(self.minTick, abs(float(size))) - 1

        return self.increment[pos]

    def incrementForSizeWithMagnifier(self, size: Decimal | float, priceMagnifier: int):
        """Return the increment size ADJUSTED FOR MAGNIFIER"""
        # All this extra work is just to compensate for potentially dynamic
        # 'priceMagnifier' rules across different symbols using the same underlying
        # marketRuleIds. Not sure if it happens, but at least this is always correct now.

        # short circuit common case
        if priceMagnifier == 1:
            return self.incrementForSize(size)

        return (self.incrementForSize(size) * priceMagnifier).normalize()

    def decimalsForIncrement(self, increment: Decimal):
        """Count length of decimal precision for input 'increment'"""
        # convince mypy this exponent is a number and not any of: ("Literal['n', 'N', 'F'] | int")
        exponent = increment.as_tuple().exponent
        assert isinstance(exponent, int)

        return max(0, -exponent)

    def round(
        self,
        price: Decimal | float,
        direction: ROUND = ROUND.NEAR,
        priceMagnifier: int = 1,
    ) -> Decimal:
        # short circuit common case
        if priceMagnifier == 1:
            return roundnear(
                self.incrementForSize(price),
                price,
                direction,
                decimals=self.decimals,
            )

        # more generalizable case for arbitrary magnifiers

        # NOTE: we assume the INBOUND PRICE to this function is ALREADY IN IBKR PRICE MAGNIFIED FORMAT.
        #       (this doesn't matter to you unless you are trading things with a 0.00025 multipler IBKR quotes as 0.025, etc)
        increment = self.incrementForSizeWithMagnifier(price, priceMagnifier)

        # use our externally abstracted reliable high precision decimal rounding helper
        # (basically, it operates in "real number space" instead of "IEEE numerical float space"
        return roundnear(
            increment,
            price,
            direction,
            decimals=self.decimalsForIncrement(increment),
        )


@dataclass(slots=True)
class MarketRuleIdCache(DualCache):
    cacheName: str = "market-rule-id-cache"

    async def fetch(self, db, ruleId: int):
        assert isinstance(ruleId, int) and ruleId > 0

        who = ruleId

        # if we are already doing a lookup, wait until the others complete
        await self.locked(who)

        # if it exists, it exists. These never change, so return what we found.
        if what := self.get(who):
            return what

        # if not found, look up on disk (maybe another service populated it)
        if what := self.checkDisk(who):
            return what

        # else, if NOT found, we need to LOOK UP, CREATE, SAVE, THEN RETURN
        # LOOKUP (note: this does NOT fall under IBKR market data pacing rules, so it is safer than other operations)

        await self.lock(who)
        try:
            rules = await db.ib.reqMarketRuleAsync(who)
        finally:
            # release the lookup lock even if there's an API exception.
            # Note: due to this being async but not parallel, this unlock doesn't notify other waiting
            #       coroutines until this method returns, so unlocking here before we actually set
            #       the data into the local lookup dicts is fine.
            self.unlock(who)

        # CREATE
        # Rules are a list of [low, increment] objects like:
        # example a: [ib_async.objects.PriceIncrement(lowEdge=0.0, increment=0.01)]
        # example b: [ib_async.objects.PriceIncrement(lowEdge=0.0, increment=0.01),
        #             ib_async.objects.PriceIncrement(lowEdge=3.0, increment=0.05)]
        # We deconstruct the input into two lists so we can search/match easier on the edge entry points.

        logger.info("[{}] Found market price rule: {}", who, rules)
        if not rules:
            # why do we sometimes get on rules results? huh? weirdos.
            return None

        # "normalize()" removes any trailing zeros so all our other math and printing works as expected
        increments = [Decimal(str(r.increment)).normalize() for r in rules]

        # find the LONGEST INCREMENT in all our potential tick sizes.
        # Market ticks can range from 0 (integer only) out to eight decimal places for things like "1/16th of 1/32th of a blah blah"
        incrementWidth = max([-int(x.as_tuple().exponent) for x in increments])

        mri = MarketRuleId(
            ruleId, tuple([r.lowEdge for r in rules]), tuple(increments), incrementWidth
        )

        # SAVE
        self[ruleId] = mri

        # RETURN
        return mri


@dataclass(slots=True)
class IInstrument:
    """icli instrument details.

    Populated by fetching live contract, instrument, and trading increment details from IBKR APIs.

    We assume this data doesn't change after we create it since none of this data is related to time.
    If you need to reset your data, clear your cache directories and everything will be re-fetched on the next startup.

    A little of everything."""

    secType: str
    tradingClass: str

    # algo names available
    # amusingly IBKR has no API to look up what the algos actually do and many of them appear
    # to be completely undocumented.
    orderTypes: frozenset[str]

    minTick: Decimal
    minSize: Decimal
    sizeIncrement: float
    suggestedSizeIncrement: float

    # priceMaginifer is IBKR speak for "adjusting the actual market quotes for easier reading."
    # For example /LE is quoted in "cents per pound" as 0.00025, but IBKR places orders as
    # "dollars per pound" or 0.025, so IBKR uses a priceMagnifer=100 here to compensate against
    # "the market" because you place orders as X.XXX and IBKR converts to (X.XXX * 100) for you.
    # Long story short: priceMagnifier GROWS the quoted precision by its "magnify" factor.
    priceMagnifier: int

    # exchange names available
    exchanges: dict[str, MarketRuleId] = field(default_factory=dict)

    # just a small local cache to avoid extra steps of lookups.
    decimalmemo: dict[Any, Any] = field(default_factory=dict)

    def round(
        self, exchange: str, price: float | Decimal, direction: ROUND = ROUND.NEAR
    ):
        """Round 'price' to valid increment based on 'exchange' requested"""

        # we assume you're passing in good exchanges here. if not, it's your own fault.
        return self.exchanges[exchange].round(
            price, direction=direction, priceMagnifier=self.priceMagnifier
        )

    def decimals(self, exchange: str) -> int:
        """Return number of decimal places required for quoting or ordering this product."""

        # priceMagnifier is an integer multiple which we GROW the underlying market tick sizes for
        # local ordering and reporting.
        # Example: minTick=0.00025 with priceMagnifier=100 means we actually use realized minTick=0.025 (0.00025 * 100),
        #          which is the same as taking 6 original digits and subtracting the 3 digits of the multiplier.

        # short circuit repeated lookups
        if found := self.decimalmemo.get(exchange):
            return found

        # short circut common case
        if self.priceMagnifier == 1:
            digits = self.exchanges[exchange].decimals
        else:
            # this looks like an annoying amount of work (and this gets called for _every_ symbol for _every_ toolbar
            # quote refresh and for every order, so this is potentially called hundreds of times per second), so
            # build-then-cache locally forever.
            pmt = Decimal(str(self.priceMagnifier)).as_tuple()
            assert pmt.exponent == 0

            # subtract 1 because priceMagnifier=1 is default, but we don't want to subtract length 1.
            # the whole 0.005 * 100 means it removes _two_ places, etc.
            magnifierDigits = len(pmt.digits) - 1

            digits = self.exchanges[exchange].decimals - magnifierDigits

        self.decimalmemo[exchange] = digits
        return digits

    @classmethod
    async def create(cls, db, contract: Contract):
        """Create IInstrument for provided contract.

        We pass in an instance of IInstrumentDatabase so this creation can access the persistent
        caches as well as run its own live IBKR metadata fetches.

        Contracts look like:
             contract=ib_async.contract.Contract(
                 secType='OPT',
                 conId=677414875,
                 symbol='NVDA',
                 lastTradeDateOrContractMonth='20240816',
                 strike=100.0,
                 right='P',
                 multiplier='100',
                 exchange='SMART',
                 currency='USD',
                 localSymbol='NVDA  240816P00100000',
                 tradingClass='NVDA'
             )

         Details look like:
             ib_async.contract.ContractDetails(
                 contract=ib_async.contract.Contract(
                     secType='OPT',
                     conId=677414875,
                     symbol='NVDA',
                     lastTradeDateOrContractMonth='20240816',
                     strike=100.0,
                     right='P',
                     multiplier='100',
                     exchange='SMART',
                     currency='USD',
                     localSymbol='NVDA  240816P00100000',
                     tradingClass='NVDA'
                 ),
                 marketName='NVDA',
                 minTick=0.01,
                 orderTypes='ACTIVETIM,AD,ADJUST,ALERT,ALGO,ALLOC,AON,AVGCOST,BASKET,COND,'
                     'CONDORDER,DAY,DEACT,DEACTDIS,DEACTEOD,DIS,FOK,GAT,GTC,GTD,GTT,HID,'
                     'ICE,IOC,LIT,LMT,MIT,MKT,MTL,NGCOMB,NONALGO,OCA,OPENCLOSE,PAON,'
                     'PEGMIDVOL,PEGMKTVOL,PEGPRMVOL,PEGSRFVOL,POSTONLY,PRICECHK,REL,'
                     'RELPCTOFS,RELSTK,SCALE,SCALERST,SIZECHK,SMARTSTG,SNAPMID,SNAPMKT,'
                     'SNAPREL,STP,STPLMT,TRAIL,TRAILLIT,TRAILLMT,TRAILMIT,VOLAT,WHATIF',
                 validExchanges='SMART,AMEX,CBOE,PHLX,PSE,ISE,BOX,BATS,NASDAQOM,CBOE2,NASDAQBX,MIAX,'
                     'GEMINI,EDGX,MERCURY,PEARL,EMERALD,MEMX,IBUSOPT',
                 priceMagnifier=1,
                 underConId=4815747,
                 longName='NVIDIA CORP',
                 contractMonth='202408',
                 industry='Technology',
                 category='Semiconductors',
                 subcategory='Electronic Compo-Semicon',
                 timeZoneId='America/New_York',
                 tradingHours='20240802:0930-20240802:1600;20240803:CLOSED;20240804:CLOSED;20240805:'
                     '0930-20240805:1600;20240806:0930-20240806:1600;20240807:0930-'
                     '20240807:1600',
                 liquidHours='20240802:0930-20240802:1600;20240803:CLOSED;20240804:CLOSED;20240805:'
                     '0930-20240805:1600;20240806:0930-20240806:1600;20240807:0930-'
                     '20240807:1600',
                 aggGroup=2,
                 underSymbol='NVDA',
                 underSecType='STK',
                 marketRuleIds='32,109,109,109,109,109,109,109,32,109,32,109,109,109,109,109,109,109,'
                     '32',
                 realExpirationDate='20240816',
                 lastTradeTime='16:00:00',
                 minSize=1.0,
                 sizeIncrement=1.0,
                 suggestedSizeIncrement=1.0
             )

             The "marketRuleIds" are their own additional data format, and the numbers are just integer id primary
             keys to use as lookups into another IBKR data fetch which _then_ returns the actual marketRules.

             marketRuleId output looks like just duplicates because they are actually paired 1-to-1 with the validExchanges
             list (each exchange can have its own "market rules" for required price increments per instrument).
        """

        try:
            # Note: this is assuming you are already passing in a very narrow already fully qualified contract.
            # If you attempt to request bulk details from partial or unqualified contracts, this could generate
            # IBKR data pacing exceptions (which is never a good thing).
            (detail,) = await asyncio.wait_for(
                db.ib.reqContractDetailsAsync(contract), timeout=5
            )
        except:
            logger.error("[{}] Failed to find contract details?", contract)
            return None

        orderTypes = frozenset(detail.orderTypes.split(","))

        priceMagnifier = detail.priceMagnifier

        # NOTE: minTick is the EXCHANGE tick size, but the IBKR tick size is (.minTick * .priceMagnifier)
        minTick = Decimal(str(detail.minTick))

        minSize = detail.minSize
        sizeIncrement = detail.sizeIncrement
        suggestedSizeIncrement = detail.suggestedSizeIncrement

        # NOTE: not all symbols populate .tradingClass
        tradingClass = contract.tradingClass

        async def buildExchangeToTickMapping():
            """Build dict of {EXCHANGE_NAME: [MarketRuleId]}"""

            # split rule id string into integers
            ridsAll = [int(x) for x in detail.marketRuleIds.split(",")]

            # split exchanges into list
            exchanges = detail.validExchanges.split(",")

            # resolve each rule id from its integer id into actual rule details
            rules = [await db.ridCache.fetch(db, rid) for rid in ridsAll]

            # if rules failed, we need to try again later
            if not all(rules):
                return None

            # create dict of {EXCHANGE: RULES} mapping
            assert len(exchanges) == len(rules)
            exchangeRuleMapping = dict(zip(exchanges, rules))

            return exchangeRuleMapping

        exchanges = await buildExchangeToTickMapping()

        # if exchanges failed, we can't generate a valid object yet...
        if not exchanges:
            # logger.warning("[{}] Failed to load exchanges?", contract)
            return None

        # provide a newly fully populated IInstrument object
        return cls(
            secType=contract.secType,
            tradingClass=tradingClass,
            orderTypes=orderTypes,
            minTick=minTick,
            minSize=minSize,
            priceMagnifier=priceMagnifier,
            sizeIncrement=sizeIncrement,
            suggestedSizeIncrement=suggestedSizeIncrement,
            exchanges=exchanges,
        )


@dataclass(slots=True)
class IInstrumentCache(DualCache):
    cacheName: str = "iinstrument-cache"

    def lookup(self, contract: Contract) -> IInstrument | None:
        """Attempt a direct lookup or return nothing.

        Note: We aren't using .get() for this because .get() is using diskcache.get,
              and we want to consume a contract here, generate a key, then do the lookup."""

        who = (
            contract.secType,
            contract.tradingClass
            or contract.symbol
            or contract.localSymbol
            or contract.conId,
        )

        return self.get(who)

    async def fetch(self, db, contract: Contract) -> IInstrument | None:
        assert isinstance(contract, Contract)

        # some instruments don't have .tradingClass, so we have to compensate otherwise)
        who = (
            contract.secType,
            contract.tradingClass
            or contract.symbol
            or contract.localSymbol
            or contract.conId,
        )

        # check if this is being locked, if so wait until it's complete, if not just fall through.
        await self.locked(who)

        # if it exists, it exists. These never change, so return what we found.
        if what := self.get(who):
            return what

        # We cannot create BAG instrument types, so give up and try something else.
        if isinstance(contract, Bag):
            return None

        # if not found, look up on disk (maybe another service populated it)
        if what := self.checkDisk(who):
            return what

        # else, if NOT found, we need to CREATE, SAVE, THEN RETURN
        # CREATE
        await self.lock(who)
        try:
            if inst := await IInstrument.create(db, contract):
                # SAVE
                self[who] = inst

                # RETURN
                return inst
        finally:
            # data is now (hopefully) available for the other readers to access
            self.unlock(who)

        # something didn't work up there...
        return None


@dataclass(slots=True)
class IInstrumentDatabase:
    """Manage looking up symbols to instrument details with caching and dynamic lookups where needed."""

    # a pointer back to our IBKRCmdlineApp instance so we can use live API services from here.
    # Note: this isn't typed as 'IBKRCmdlineApp' because we import instrumentdb _from_ there too.
    state: Any

    # just a shorthand pointer from state.ib
    ib: IB = field(init=False)

    # cache of (secType, tradingClass or symbol) to its actual trading details.
    # Note: 'Symbol' here is the _underlying_ symbol if this is options or futures because we don't
    #       need to cache each unique 'localSymbol' because market rules are common based on underlying symbols.
    symbolToTradingClassMap: dict[tuple[str, str], IInstrument] = field(
        default_factory=dict
    )

    ridCache: MarketRuleIdCache = field(default_factory=MarketRuleIdCache)
    iiCache: IInstrumentCache = field(default_factory=IInstrumentCache)

    asyncTaskLocker: dict[Contract, Coroutine] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Just extract IB into a local instance variable so we can use it easier
        self.ib = self.state.ib

    async def round(
        self, contract: Contract, price: Decimal | float, direction: ROUND = ROUND.NEAR
    ) -> Decimal | None:
        """Return an instrument-correct value given a contract and price.

        This is important because different instruments have different price levels.
        For example, stocks are quoted in cents while interest rates are quoted down to 0.001
        while Forex is often quoted down to 0.0001 etc.

        Proper rounding and decimal precision reporting requires knowing the exchange and instrument
        being traded, so we have this complex multi-hierarchy cache of the actual data requirements."""

        # if bag, replace input contract with contract for first leg of spread so we can fetch rounding details.
        # TODO: the "how to round a bag" logic should be cleaned up more if we have e.g. multi-asset-bags, but
        #       for now we assume bags are just spreads of the same instrument type and underlying.
        #       e.g. what is the proper round for Bag([AAPL STOCK 100], [AAPL OPTION 1]) etc?
        if isinstance(contract, Bag):
            # SPECIAL OVERRIDE: /VX bags (calendar spreads) operate in 0.01 increments: https://www.cboe.com/tradable_products/vix/vix_futures/specifications/
            # (Also match for SPX/SPXW behavior where spreads operate in 0.01 increments even though individual legs are priced in 0.05 increments)
            if contract.symbol == "VIX" and contract.exchange == "CFE":
                return roundnear(
                    Decimal("0.01"), Decimal(str(price)), direction, decimals=2
                )

            (contract,) = await self.state.qualify(
                Contract(conId=contract.comboLegs[0].conId)
            )

        if found := await self.iiCache.fetch(self, contract):
            return found.round(contract.exchange, price, direction)

        logger.error(
            "Failed to find or create instrument? Asked for: {} :: {}",
            contract.exchange,
            contract,
        )

        return None

    def roundOrNothing(
        self, contract: Contract, price: Decimal | float
    ) -> float | None:
        """Helper for places we can't use the async round implementation.

        Basically checks if we CAN round, then we round-to-float, else we
        schedule an async background fetch of the data and return None, but
        soon the data will exist for future lookups.

        This is primarily intended for utility/quote/check interfaces and
        not direct trade data manipulation."""

        if found := self.iiCache.lookup(contract):
            return float(found.round(contract.exchange, price))

        # if this is a bag, we need to look up EACH LEG to find the LEAST COMMON MIN TICK
        if isinstance(contract, Bag):
            # Simple (lazy) way for now:
            # Step 1: fetch first contract by conId from CONTRACT CACHE (must ALREADY EXIST in cache since we don't fetch here)
            # Step 2: run round on first contract. Done.
            leg0 = self.state.conIdCache.get(contract.comboLegs[0].conId)
            return self.roundOrNothing(leg0, price)

            # Correct (perfect) way eventually:
            # Step 1: fetch all contracts by conid from CONTRACT CACHE
            # Step 2: fetch IInstrument objects for each contract
            # Step 3: find minimum common denominator for all legs given the exchange and price request

        # logger.warning("[{}] Instrument not populated yet. Scheduling for lookup.", contract)

        # schedule this instrument cache to populate in the background...
        asyncio.create_task(self.iiCache.fetch(self, contract))

        return None

    def exchanges(self, contract: Contract) -> frozenset[str]:
        if found := self.iiCache.lookup(contract):
            return frozenset(found.exchanges.keys())

        # if can't find in cache, so just assume the contract itself is true
        return frozenset([contract.exchange])

    def orderTypes(self, contract: Contract) -> frozenset[str]:
        if found := self.iiCache.lookup(contract):
            return found.orderTypes

        # if can't find in cache, we can't verify anything.
        return frozenset()

    def exchangesCount(self, contract: Contract) -> int:
        if found := self.iiCache.lookup(contract):
            return len(found.exchanges)

        # assume anything else is singular-exchange routed (like bags are SMART-only, etc)
        return 1

    def decimals(self, contract) -> int | None:
        """Return number of decimal places required for this symbol to be properly quoted and ordered."""
        if found := self.iiCache.lookup(contract):
            return found.decimals(contract.exchange)

        # logger.warning("[{}] Instrument not populated yet. Scheduling for lookup.", contract)

        # schedule this instrument cache to populate in the background...
        asyncio.create_task(self.iiCache.fetch(self, contract))

        return None

    def load(self, *contracts) -> None:
        """Helper method to verify contracts exist in DB and create them if they don't exist."""
        for contract in contracts:
            # if it doesn't currently exist in our system
            if contract and not self.iiCache.lookup(contract):
                # schedule it to be created soon
                asyncio.create_task(self.iiCache.fetch(self, contract))
