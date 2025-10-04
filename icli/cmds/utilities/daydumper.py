"""Command: daydumper

Category: Utilities
"""

import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ib_async import (
    Contract,
)
from loguru import logger
from mutil.dispatch import DArg
from mutil.frame import printFrame

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import copy
import math

import pandas as pd
import prettyprinter as pp  # type: ignore


@command(names=["daydumper"])
@dataclass
class IOpDayDumper(IOp):
    """Save bar history for a symbol to disk using IBKR data APIs."""

    symbol: str = field(init=False)
    back: int = field(init=False)
    interval: str = field(init=False)

    def argmap(self):
        return [
            DArg("symbol", convert=str.upper),
            DArg("back", convert=int, default=7),
            DArg("interval", default="1 day"),
        ]

    async def run(self):
        foundSymbol, originalContract = await self.state.positionalQuoteRepopulate(
            self.symbol
        )

        if not foundSymbol:
            logger.error("Symbol not found? Can't perform a lookup!")
            return

        self.symbol = foundSymbol
        assert originalContract

        # For futures, use continuous full historical representation instead of a local expiration date
        # (also, do this on a COPY because we don't want to overwite the shared/cached contract object)
        contract: Contract
        if isinstance(originalContract, Future):
            contract = copy.copy(originalContract)
            contract.secType = "CONTFUT"
        else:
            contract = originalContract

        # fetch data
        found = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            # IBKR only accepts 'D' requests for <= 365 days, else it requires full years of retrieval
            durationStr=f"{self.back} D"
            if self.back <= 365
            else f"{math.ceil(self.back / 365)} Y",
            # Valid bar sizes are:
            # 1 secs	5 secs	10 secs	15 secs	30 secs 1 min	2 mins	3 mins	5 mins	10 mins	15 mins	20 mins	30 mins 1 hour	2 hours	3 hours	4 hours	8 hours 1 day 1 week 1 month
            barSizeSetting=self.interval,
            whatToShow="TRADES",
            useRTH=False,
            keepUpToDate=False,
            timeout=6,
        )

        # augment data a little

        def wwma(values, n):
            return values.ewm(
                alpha=1 / n, min_periods=n, ignore_na=True, adjust=False
            ).mean()

        def atr(df, n=14):
            data = df.copy()
            high = data.high
            low = data.low
            close = data.close
            data["tr0"] = abs(high - low)
            data["tr1"] = abs(high - close.shift())
            data["tr2"] = abs(low - close.shift())
            tr = data[["tr0", "tr1", "tr2"]].max(axis=1)
            atr = wwma(tr, n)
            return atr

        def bollinger_zscore(series: pd.Series, length: int = 20) -> pd.Series:
            # Ref: https://stackoverflow.com/a/77499303/
            rolling = series.rolling(length)
            mean = rolling.mean()
            std = rolling.std(ddof=0)
            return (series - mean) / std

        def bollinger_bands(
            series: pd.Series,
            length: int = 20,
            *,
            num_stds: tuple[float, ...] = (2, 0, -2),
            prefix: str = "",
        ) -> pd.DataFrame:
            # Ref: https://stackoverflow.com/a/74283044/
            rolling = series.rolling(length)
            bband0 = rolling.mean()
            bband_std = rolling.std(ddof=0)
            return pd.DataFrame(
                {
                    f"{prefix}{num_std}": (bband0 + (bband_std * num_std))
                    for num_std in num_stds
                }
            )

        # save data
        digits: Final = self.state.decimals(originalContract)
        table = pd.DataFrame(found).convert_dtypes()

        if table.empty:
            logger.error("No result?")
            return

        table["diff"] = table.close.diff(periods=1)

        for n in [5, 10, 20, 30, 60, 220, 325]:
            field = f"sma_{n}"
            table[field] = round(table.close.rolling(n).mean(), digits)
            table[f"{n}±"] = table.apply(
                lambda row: "+"
                if row.close > row[field]
                else "-"
                if row.close < row[field]
                else "=",
                axis=1,
            )

        table = pd.concat([table, bollinger_bands(table.close, prefix="bb_")], axis=1)
        table["bb_zscore"] = bollinger_zscore(table.close)
        table["range"] = table.high - table.low
        table["atr"] = atr(table, 6)

        table = round(table, digits)

        # saving requires removing the timezone, so keep time and just un-localize it.
        try:
            # Note: tz_convert(None) moves it to UTC. tz_localize(None) leaves the time unchanged but drops the timezone from the object.
            table.date = table.date.dt.tz_localize(None)
        except:
            # if we have a full datetime, the above works, else it fails, but failure is okay here.
            pass

        logger.info("{}", pp.pformat(contract))
        printFrame(table, f"{self.symbol} History")

        duration_map: Final = {
            "1 secs": 1,
            "5 secs": 5,
            "10 secs": 10,
            "15 secs": 15,
            "30 secs": 30,
            "1 min": 60,
            "2 mins": 120,
            "3 mins": 180,
            "5 mins": 300,
            "10 mins": 600,
            "15 mins": 900,
            "20 mins": 1200,
            "30 mins": 1800,
            "1 hour": 3600,
            "2 hours": 7200,
            "3 hours": 10800,
            "4 hours": 14400,
            "8 hours": 28800,
            "1 day": 86400,
            "1 week": 604800,
            "1 month": 2592000,
        }

        barDiffSec = duration_map[self.interval]
        logger.info("Bar length: {} ({} seconds)", self.interval, barDiffSec)

        # name output as {localSymbol}.{barDuration}.table.json
        # TODO: make the storage directory a parameter/envvar somewhere
        where = pathlib.Path("bardb")
        where.mkdir(parents=True, exist_ok=True)

        printName = self.state.nameForContract(contract).replace(" ", "_")

        # we also have to strip slashes in futures names so they don't turn into directory paths...
        filename = where / f"{printName.replace('/', ',')}.{barDiffSec}.table.json"
        table.to_json(filename, orient="table")

        # done
        logger.info("Saved to: {}", filename)

        lb = LevelBreacher(barDiffSec)

        # we store open/high/low/close values _directly_ but we use 'sma' as a prefix matcher with sub-lookback-duration fields.
        populate: Final = "open high low close average sma bb_2 bb_0 bb_-2".split()

        # use last row of dataframe (with most recently updated SMA values)
        lastlast = table.iloc[-2]
        last = table.iloc[-1]

        # create level matchers for each data field we're interested in tracking
        # (by iterating the name/field pairs of the last row created)
        levels = []

        def buildLevelsFromRow(row, desc, populateSMA=True, isToday=False):
            for name, level in row.items():
                assert isinstance(name, str)

                # if value is NaN, don't generate a level checker for this component
                if level != level:
                    continue

                for p in populate:
                    if name.startswith(p):
                        s = name.split("_")

                        # the lookback duration is EITHER a key split like 'sma_220' — OR — the direct bar interval if no sma is present.
                        # (e.g. 1 day bars with "open" tracking is "/ES DOWN open 1 day")
                        if len(s) > 1:
                            # SMA split durations are just their own size with no more details
                            lookback = int(s[1])
                            lookbackName = s[1]
                        else:
                            # else, the duration is the BAR duration, which we calculate back to whole values of minute/day/week/month/year
                            lookback = barDiffSec
                            lookbackName = convert_time(barDiffSec)

                        # SMA levels are optional because we don't want to, for example, populate *yesterday* SMA values
                        if p == "sma" and populateSMA:
                            l = LevelLevels(p, lookback, lookbackName, level)
                            levels.append(l)
                        elif p != "sma":
                            # only populate open/high/low/close/average values if we are on 30 minute bars or larger
                            # (because it's too nosiy bouncing around "last seen 2 minute close breach" etc)
                            if barDiffSec >= 1800:
                                # if bar is TODAY (meaning bar is STILL OPEN and changing close/high/low/average), then don't
                                # populate the changing fields.
                                # (Note: for TODAY we _do_ accept the current bb std ranges from the active 'close' math reported... because why not)
                                if isToday and (p != "open" and not p.startswith("bb")):
                                    continue

                                l = LevelLevels(
                                    f"{p} {desc}", lookback, lookbackName, level
                                )
                                levels.append(l)

        # date math is hacky here (isn't it always?) because last.date could be a Date _or_ a Timestamp, so we always re-wrap it in a Timestamp so the compare can work
        today = pd.Timestamp.now().floor("D")
        lastIsToday: Final = pd.Timestamp(last.date) >= today
        lastLastIsToday: Final = pd.Timestamp(lastlast.date) >= today

        # for TODAY (if row[-1] date is >= CALENDAR TODAY) we want to ignore close (random) and high/low (caught by agent-sever)
        # Note: meaning of 'today' is the last row, so 'today' is only 'today' if the stats are generated on the current trading day.
        # also, only populate the "yesterday" value if the iloc[-2] bar is NOT today.
        if barDiffSec >= 300:
            buildLevelsFromRow(
                lastlast, "previous", populateSMA=False, isToday=lastLastIsToday
            )

        buildLevelsFromRow(last, "today", isToday=lastIsToday)

        lb = LevelBreacher(barDiffSec, levels)

        symkey = lookupKey(contract)
        logger.info(
            "[{}] Created Level Breacher alerting hierarchy:\n{}",
            printName,
            pp.pformat(lb),
        )

        # a little abstraction breakage here. Is fine.
        self.state.quoteState[symkey].levels[barDiffSec] = lb
