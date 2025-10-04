"""Command: executions

Category: Portfolio
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg
from mutil.frame import printFrame
from mutil.numeric import fmtPrice

from icli.cmds.base import IOp, command
from icli.cmds.utils import (
    addRowSafe,
)
from icli.helpers import *

if TYPE_CHECKING:
    pass
import pandas as pd
import prettyprinter as pp  # type: ignore
import pytz


@command(names=["executions"])
@dataclass
class IOpExecutions(IOp):
    """Display all executions including commissions and PnL."""

    symbols: set[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*symbols",
                desc="Optional symbols to filter for in result listings",
                convert=lambda x: set([s.upper() for s in x]),
            )
        ]

    async def run(self):
        # "Fills" has:
        # - contract
        # - execution (price at exchange)
        # - commissionReport (commission and PNL)
        # - time (UTC)
        # .executions() is the same as just the 'execution' value in .fills()

        # the live updating only works for the current client activity, so to fetch _all_
        # executions for the entire user over all clients, run "exec refresh" or "exec update"
        # ALSO NOTE: we stopped fetching executions on startup too, so this will restore previous
        #            executions if you restart your client in the middle of a trading session.
        REFRESH_CHECK = {":REFRESH", ":UPDATE", ":U", ":R", ":UPD", ":REF"}
        SELF_CHECK = {":SELF", ":MINE", ":S", ":M"}

        assert (
            REFRESH_CHECK & SELF_CHECK == set()
        ), "There are conflicting keywords in REFRESH_CHECK vs SELF_CHECK?"

        if REFRESH_CHECK & self.symbols:
            self.symbols -= REFRESH_CHECK
            await self.state.loadExecutions()

        fills = self.ib.fills()
        # logger.info("Fills: {}", pp.pformat(fills))
        contracts = []
        executions = []
        commissions = []
        for f in fills:
            contracts.append(f.contract)
            executions.append(f.execution)
            commissions.append(f.commissionReport)

        if False:
            logger.info(
                "C: {}\nE: {}: CM: {}",
                pp.pformat(contracts),
                pp.pformat(executions),
                pp.pformat(commissions),
            )

        use = []
        for name, l in [
            ("Contracts", contracts),
            ("Executions", executions),
            ("Commissions", commissions),
        ]:
            df = pd.DataFrame(l)  # type: ignore
            if df.empty:
                logger.info("No {}", name)
            else:
                use.append((name, df))

        if not use:
            return None

        df = pd.concat({name: frame for name, frame in use}, axis=1)

        # Goodbye multiindex...
        df.columns = df.columns.droplevel(0)

        # enforce the dataframe is ordered from oldest executions to newest executions as defined by full original timestamp order.
        df.sort_values(by=["time", "clientId"], inplace=True)

        # show only executions for the current client id
        # TODO: allow showing only from a specific client id provided as a paramter too?
        if SELF_CHECK & self.symbols:
            self.symbols -= SELF_CHECK
            df = df[df.clientId == self.state.clientId]

        # if symbol filter requested, remove non-matching contracts.
        # NOTE: we filter on SYMBOL and not "localSymbol" so we don't currently match on extended details like OCC symbols.
        if self.symbols:
            df = df[df.symbol.isin(self.symbols)]

        # clean up any non-matching values due to symbols filtering
        if self.symbols:
            df = df[df.conId.notna()]

        if df.empty:
            logger.info(
                "[{}] No execution history found!", " ".join(sorted(self.symbols))
            )
            return False

        # Remove duplicate columns...
        df = df.loc[:, ~df.columns.duplicated()]

        # convert to Eastern time and drop date (since these reports only show up for one day, it's all duplicate details)
        df["time"] = df["time"].apply(pd.Timestamp).dt.tz_convert("US/Eastern")
        df["timestamp"] = df["time"]

        df["date"] = df["time"].dt.strftime("%Y-%m-%d")
        df["time"] = df["time"].dt.strftime("%H:%M:%S")

        df["c_each"] = df.commission / df.shares

        tointActual = ["clientId", "orderId"]
        df[tointActual] = df[tointActual].map(lambda x: f"{int(x)}" if x else "")

        # really have to stuff this multiplier change in there due to pandas typing requirements
        df.multiplier = df.multiplier.replace("", 1).fillna(1).astype(float)

        df["total"] = round(df.shares * df.avgPrice, 2) * df.multiplier

        # Note: 'realizedPNL' for the closing transactions *already* includes commissions for both the buy and sell executions,
        #       so *don't* subtract commissions again anywhere.
        df["dayProfit"] = df.realizedPNL.cumsum()

        df["RPNL_each"] = df.realizedPNL / df.shares

        # provide a weak calculation of commission as percentage of PNL and of traded value.
        # Note: we estimate the entry commission by just doubling the exit commission (which isn't 100% accurate, but
        #       if the opening trade was more than 1 day ago, we don't have the opening matching executions to read
        #       the execution from (and we aren't keeping a local fullly logged execution history, but maybe we should
        #       add logged execution history as a feature in the future?)
        df["c_pct"] = df.commission / (df.total + df.commission)
        df["c_pct_RPNL"] = (df.commission * 2) / (df.realizedPNL + (df.commission * 2))

        dfByTrade = df.groupby("orderId side localSymbol".split()).agg(  # type: ignore
            dict(
                date=["min"],
                time=[("start", "min"), ("finish", "max")],  # type: ignore
                price=["mean"],
                shares=["sum"],
                total=["sum"],
                commission=["sum"],
                # TODO: we need to calculate a manlal c_each by (total commision / shares) instead of mean of the c_each directly
                c_each=["mean"],
            )
        )

        # also show if the order occurred via multiple executions over time
        # (single executions will have zero duration, etc)
        dfByTrade["duration"] = pd.to_datetime(
            dfByTrade["time"]["finish"], format="%H:%M:%S"
        ) - pd.to_datetime(dfByTrade["time"]["start"], format="%H:%M:%S")

        # convert the default pandas datetime difference just into a number of seconds per row
        # (the returned "Series" from the subtraction above doesn't allow .seconds to be applied
        #  as a column operation, so we apply it row-element-wise here instead)
        dfByTrade["duration"] = dfByTrade.duration.apply(lambda x: x.seconds)

        # also show commission percent for traded value per row
        dfByTrade["c_pct"] = dfByTrade.commission / (
            dfByTrade.total + dfByTrade.commission
        )

        dfByTimeProfit = df.copy().sort_values(
            by=["time", "orderId", "secType", "side", "localSymbol"]
        )

        needsPrices = "price shares total commission c_each".split()
        dfByTrade[needsPrices] = dfByTrade[needsPrices].map(fmtPrice)

        # this currently has a false pandas warning about "concatenation with empty or all-NA entries is deprecated"
        # but nothing is empty or NA in these columns. Their logic for checking their warning condition is just broken.
        # (or their "FutureWarning" error message is so bad we can't actually see what the problem is)
        df = addRowSafe(df, "sum", df[["shares", "price", "commission", "total"]].sum())

        df = addRowSafe(
            df,
            "sum-buy",
            df[["shares", "price", "commission", "total"]][df.side == "BOT"].sum(),
        )

        df = addRowSafe(
            df,
            "sum-sell",
            df[["shares", "price", "commission", "total"]][df.side == "SLD"].sum(),
        )

        df.loc["profit", "total"] = (
            df.loc["sum-sell"]["total"] - df.loc["sum-buy"]["total"]
        )
        df.loc["profit", "price"] = (
            df.loc["sum-sell"]["price"] - df.loc["sum-buy"]["price"]
        )

        eachSharePrice = ["c_each", "shares", "price"]
        df = addRowSafe(df, "med", df[eachSharePrice].median())
        df = addRowSafe(df, "mean", df[eachSharePrice].mean())

        needsPrices = "c_each shares price avgPrice commission realizedPNL RPNL_each total dayProfit c_pct c_pct_RPNL".split()
        df[needsPrices] = df[needsPrices].map(fmtPrice)

        # convert contract IDs to integers (and fill in any missing
        # contract ids with placeholders so they don't get turned to
        # strings with the global .fillna("") below).
        df.conId = df.conId.fillna(-1).astype(int)

        # new pandas strict typing doesn't allow numeric columns to become "" strings, so now just
        # convert ALL columns to a generic object type
        df = df.astype(object)

        # display anything NaN as empty strings so it doesn't clutter the interface
        df.fillna("", inplace=True)

        df.rename(columns={"lastTradeDateOrContractMonth": "conDate"}, inplace=True)
        # ignoring: "execId" (long string for execution recall) and "permId" (???)

        # removed: lastLiquidity avgPrice
        df = df[
            (
                """clientId secType conId symbol conDate right strike date exchange tradingClass localSymbol time orderId
         side  shares  cumQty price    total realizedPNL RPNL_each
         commission c_each c_pct c_pct_RPNL dayProfit""".split()
            )
        ]

        dfByTimeProfit.set_index("timestamp", inplace=True)

        dfByTimeProfit["profit"] = dfByTimeProfit.where(dfByTimeProfit.realizedPNL > 0)[
            "realizedPNL"
        ]

        dfByTimeProfit["loss"] = dfByTimeProfit.where(dfByTimeProfit.realizedPNL < 0)[
            "realizedPNL"
        ]

        # using "PNL == 0" to determine an open order should work _most_ of the time, but if you
        # somehow get exactly a $0.00 PNL on a close, it will be counted incorrectly here.
        dfByTimeProfit["opening"] = dfByTimeProfit.where(
            dfByTimeProfit.realizedPNL == 0
        )["orderId"]
        dfByTimeProfit["closing"] = dfByTimeProfit.where(
            dfByTimeProfit.realizedPNL != 0
        )["orderId"]

        # ==============================================
        # Profit by Half Hour (split out By Day) Control
        # ==============================================

        # Function to assign each timestamp to its corresponding trading day
        # Trading day is defined as 6PM ET to 5PM ET the next day
        def assign_trading_day(timestamp):
            # Convert timestamp to Eastern Time if it's not already
            if (
                timestamp.tzinfo is None
                or timestamp.tzinfo.utcoffset(timestamp) is None
            ):
                # Assumes the timestamp is already in ET, adjust if needed
                et_timestamp = timestamp
            else:
                # If timestamp has timezone info, convert to ET
                eastern = pytz.timezone("US/Eastern")
                et_timestamp = timestamp.astimezone(eastern)

            # If time is before 6PM, it belongs to the previous day's trading session
            if et_timestamp.hour < 18:  # Before 6PM ET
                trading_day = et_timestamp.date()
            else:  # 6PM ET or later
                trading_day = et_timestamp.date() + pd.Timedelta(days=1)

            return pd.Timestamp(trading_day)

        # Apply the trading day logic to your dataframe
        dfByTimeProfit["trading_day"] = dfByTimeProfit.index.map(assign_trading_day)

        # Group by trading day and then resample by 30 minutes
        trading_day_groups = dfByTimeProfit.groupby("trading_day")

        # Create an empty dataframe to store results
        all_results = pd.DataFrame()

        # Process each trading day separately, printing one table per trading day
        desc = ""
        if self.symbols:
            desc = f" Filtered for: {', '.join(self.symbols)}"

        # Print the original execution summary
        printFrame(df.convert_dtypes(), f"Execution Summary{desc}")

        # Create the daily summary table to show at the end
        trading_day_summary = dfByTimeProfit.groupby("trading_day").agg(
            dict(
                realizedPNL="sum",
                orderId=[("orders", "nunique"), ("executions", "count")],
                profit="count",
                loss="count",
                opening="nunique",
                closing="nunique",
            )
        )

        # Process each trading day and print tables individually
        for trading_day, group in trading_day_groups:
            # Resample the group by 30 minutes
            day_profit = group.resample("30Min").agg(
                dict(
                    realizedPNL="sum",
                    orderId=[("orders", "nunique"), ("executions", "count")],
                    profit="count",
                    loss="count",
                    opening="nunique",
                    closing="nunique",
                )
            )

            # Calculate cumulative sum for just this trading day
            day_profit["dayProfit"] = day_profit.realizedPNL.cumsum()

            # Format prices
            needsPrices = "realizedPNL dayProfit".split()
            day_profit[needsPrices] = day_profit[needsPrices].map(fmtPrice)

            # Add sum row for this day's table
            numeric_cols = "orderId profit loss opening closing".split()
            day_profit.loc["sum"] = day_profit.loc[:, numeric_cols].sum()

            # Format the trading day for display
            trading_day_str = pd.Timestamp(trading_day).strftime("%Y-%m-%d")

            # Print separate table for each trading day
            printFrame(
                day_profit.convert_dtypes(),
                f"Profit by Half Hour for Trading Day {trading_day_str}{desc}",
            )

        # Format the daily summary
        trading_day_summary["totalProfit"] = trading_day_summary.realizedPNL
        trading_day_summary[["realizedPNL", "totalProfit"]] = trading_day_summary[
            ["realizedPNL", "totalProfit"]
        ].map(fmtPrice)

        # Print the trading day summary at the end
        printFrame(trading_day_summary.convert_dtypes(), f"Profit by Trading Day{desc}")

        # ==============================================
        # END Profit by Half Hour (split out By Day) Control
        # ==============================================

        if False:
            # (original "profitByHour" (actually by half hour) implementation that didn't break out multiple days
            profitByHour = dfByTimeProfit.resample("30Min").agg(  # type: ignore
                dict(
                    realizedPNL="sum",
                    orderId=[("orders", "nunique"), ("executions", "count")],  # type: ignore
                    profit="count",
                    loss="count",
                    opening="nunique",
                    closing="nunique",
                )
            )

            profitByHour["dayProfit"] = profitByHour.realizedPNL.cumsum()

            needsPrices = "realizedPNL dayProfit".split()
            profitByHour[needsPrices] = profitByHour[needsPrices].map(fmtPrice)

            desc = ""
            if self.symbols:
                desc = f" Filtered for: {', '.join(self.symbols)}"

            printFrame(df.convert_dtypes(), f"Execution Summary{desc}")

            profitByHour.loc["sum"] = profitByHour.loc[
                :, "orderId profit loss opening closing".split()
            ].sum()

            printFrame(profitByHour.convert_dtypes(), f"Profit by Half Hour{desc}")

        printFrame(
            dfByTrade.sort_values(
                by=[("date", "min"), ("time", "start"), "orderId", "localSymbol"]  # type: ignore
            ).convert_dtypes(),
            f"Execution Summary by Complete Order{desc}",
        )
