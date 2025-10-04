"""Command: simulate

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ib_async import (
    CommissionReport,
    Execution,
    Fill,
    Option,
    Order,
    OrderStatus,
    Trade,
    TradeLogEntry,
)
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import random


@command(names=["simulate"])
@dataclass
class IOpSimulate(IOp):
    """Simluate events (for testing event handlers mostly)"""

    category: str = field(init=False)
    params: list[str] = field(init=False)

    def argmap(self):
        return [DArg("category", convert=str.lower), DArg("*params")]

    async def run(self):
        args = dict(zip(self.params, self.params[1:]))

        match self.category:
            case "commission":
                orderId = random.randint(0, 200_000)
                qty = float(args.get("qty", 2))
                price = float(args.get("price", 6.47))
                side = args.get("side", "SLD")
                trade = Trade(
                    contract=Option(
                        conId=650581442,
                        symbol="NVDA",
                        lastTradeDateOrContractMonth="20240920",
                        strike=100.0,
                        right="P",
                        multiplier="100",
                        exchange="SMART",
                        currency="USD",
                        localSymbol="NVDA  240920P00100000",
                        tradingClass="NVDA",
                    ),
                    order=Order(
                        orderId=orderId,
                        clientId=4,
                        permId=1139889072,
                        action="BUY",
                        totalQuantity=5.0,
                        orderType="LMT",
                        lmtPrice=5.77,
                        auxPrice=0.0,
                        tif="GTC",
                        ocaGroup="1139889071",
                        ocaType=3,
                        parentId=30741,
                        displaySize=2147483647,
                        rule80A="0",
                        openClose="",
                        volatilityType=0,
                        deltaNeutralOrderType="None",
                        referencePriceType=0,
                        account="U",
                        clearingIntent="IB",
                        adjustedOrderType="None",
                        cashQty=0.0,
                        dontUseAutoPriceForHedge=True,
                    ),
                    orderStatus=OrderStatus(
                        orderId=orderId,
                        status="SIMULATED",
                        filled=0.0,
                        remaining=5.0,
                        avgFillPrice=0.0,
                        permId=1139889072,
                        parentId=0,
                        lastFillPrice=0.0,
                        clientId=4,
                        whyHeld="",
                        mktCapPrice=0.0,
                    ),
                    fills=[],
                    log=[
                        TradeLogEntry(
                            time=datetime.datetime(
                                2024,
                                8,
                                7,
                                1,
                                47,
                                27,
                                517468,
                                tzinfo=datetime.timezone.utc,
                            ),
                            status="SIMULATED",
                            message="",
                            errorCode=0,
                        ),
                        TradeLogEntry(
                            time=datetime.datetime(
                                2024,
                                8,
                                7,
                                4,
                                16,
                                13,
                                396221,
                                tzinfo=datetime.timezone.utc,
                            ),
                            status="SIMULATED",
                            message="",
                            errorCode=0,
                        ),
                    ],
                    advancedError="",
                )
                fill = Fill(
                    contract=Option(
                        conId=691069999,
                        symbol="SPX",
                        lastTradeDateOrContractMonth="20240816",
                        strike=5520.0,
                        right="P",
                        multiplier="100",
                        exchange="CBOE",
                        currency="USD",
                        localSymbol="SPXW  240816P05520000",
                        tradingClass="SPXW",
                    ),
                    execution=Execution(
                        execId="0000f711.670b89f9.02.01.01",
                        time=datetime.datetime(
                            2024, 8, 16, 10, 8, 56, tzinfo=datetime.timezone.utc
                        ),
                        acctNumber="U",
                        exchange="CBOE",
                        side=side,
                        shares=qty,
                        price=price,
                        permId=1143027176,
                        clientId=77,
                        orderId=orderId,
                        liquidation=0,
                        cumQty=2.0,
                        avgPrice=6.47,
                        orderRef="",
                        evRule="",
                        evMultiplier=0.0,
                        modelCode="",
                        lastLiquidity=1,
                        pendingPriceRevision=False,
                    ),
                    commissionReport=CommissionReport(
                        execId="",
                        commission=random.gauss(1.24, 1.24),
                        currency="",
                        realizedPNL=0.0,
                        yield_=0.0,
                        yieldRedemptionDate=0,
                    ),
                    time=datetime.datetime(
                        2024, 8, 16, 10, 8, 56, 756793, tzinfo=datetime.timezone.utc
                    ),
                )

                # test it through the live handler callback...
                self.state.commissionHandler(trade, fill, None)
