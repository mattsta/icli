from dataclasses import dataclass
from typing import *


@dataclass
class FutureSymbol:
    symbol: str
    exchange: str
    name: str
    multiplier: Union[float, str] = ""  # ib_insync default is '' for none here
    delayed: bool = False


# When 3.10 happens:
# Symbol: TypeAlias = str
Symbol = str


def generateFuturesMapping() -> dict[Symbol, FutureSymbol]:
    """Generate mapping of:
        - future symbol => exchange, name of future
    using the IBKR margin tables (for common exchanges we want at least).

    The values should be fairly static, so we just cache a manual run as
    the global FUTS_EXCHANGE in this file."""
    import pandas as pd

    futs = pd.read_html("https://www.interactivebrokers.com/en/index.php?f=26662")  # type: ignore
    fs = {}
    for ex in futs:
        for idx, row in ex.iterrows():
            # Note: we don't use the name of the columns because the IBKR tables
            #       use different table row names per section, which is annoying,
            #       but their orders remain stable across tables.
            exchange = row[0]
            symbol = row[1]
            name = row[2]
            if exchange in {
                "ECBOT",
                "CME",
                "GLOBEX",
                "CMECRYPTO",
                "CFE",
                "NYBOT",
                "NYMEX",
                "SMFE",
                "ICECRYPTO",
            }:
                fs[row[1]] = FutureSymbol(symbol=symbol, exchange=exchange, name=name)

    return fs


# simple mapping from name of future to exchange for future.
# Used for generating ibkr api Future Contract specification since
# each symbol must have an exchange declared.
FUTS_EXCHANGE = {
    "AMB90": FutureSymbol(
        symbol="AMB90",
        exchange="CFE",
        name="CBOE Three-Month AMERIBOR Compound Average Rate Index",
        multiplier="",
        delayed=False,
    ),
    "IBXXIBHY": FutureSymbol(
        symbol="IBXXIBHY",
        exchange="CFE",
        name="iBoxx iShares $ High Yield Corporate Bond Index TR",
        multiplier="",
        delayed=False,
    ),
    "IBXXIBIG": FutureSymbol(
        symbol="IBXXIBIG",
        exchange="CFE",
        name="iBoxx iShares $ Investment Grade Corporate Bond Index TR",
        multiplier="",
        delayed=False,
    ),
    "VIX": FutureSymbol(
        symbol="VIX",
        exchange="CFE",
        name="CBOE Volatility Index",
        multiplier="",
        delayed=False,
    ),
    "VXM": FutureSymbol(
        symbol="VXM",
        exchange="CFE",
        name="Mini Cboe Volatility Index Futures",
        multiplier="",
        delayed=False,
    ),
    "ACD": FutureSymbol(
        symbol="ACD",
        exchange="GLOBEX",
        name="Australian dollar",
        multiplier="",
        delayed=False,
    ),
    "AUD": FutureSymbol(
        symbol="AUD",
        exchange="GLOBEX",
        name="Australian dollar",
        multiplier="",
        delayed=False,
    ),
    "AJY": FutureSymbol(
        symbol="AJY",
        exchange="GLOBEX",
        name="Australian dollar",
        multiplier="",
        delayed=False,
    ),
    "GBP": FutureSymbol(
        symbol="GBP",
        exchange="GLOBEX",
        name="British pound",
        multiplier="",
        delayed=False,
    ),
    "BRE": FutureSymbol(
        symbol="BRE",
        exchange="GLOBEX",
        name="Brazilian Real in US Dollars",
        multiplier="",
        delayed=False,
    ),
    "CAD": FutureSymbol(
        symbol="CAD",
        exchange="GLOBEX",
        name="Canadian dollar",
        multiplier="",
        delayed=False,
    ),
    "CZK": FutureSymbol(
        symbol="CZK",
        exchange="GLOBEX",
        name="Czech koruna",
        multiplier="",
        delayed=False,
    ),
    "EUR": FutureSymbol(
        symbol="EUR",
        exchange="GLOBEX",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "ECK": FutureSymbol(
        symbol="ECK",
        exchange="GLOBEX",
        name="Czech koruna",
        multiplier="",
        delayed=False,
    ),
    "GE": FutureSymbol(
        symbol="GE", exchange="GLOBEX", name="EURODOLLARS", multiplier="", delayed=False
    ),
    "EHF": FutureSymbol(
        symbol="EHF",
        exchange="GLOBEX",
        name="Hungarian forint",
        multiplier="",
        delayed=False,
    ),
    "EM": FutureSymbol(
        symbol="EM",
        exchange="GLOBEX",
        name="1 Month LIBOR (Int. Rate)",
        multiplier="",
        delayed=False,
    ),
    "EPZ": FutureSymbol(
        symbol="EPZ",
        exchange="GLOBEX",
        name="Polish zloty",
        multiplier="",
        delayed=False,
    ),
    "GF": FutureSymbol(
        symbol="GF",
        exchange="GLOBEX",
        name="Feeder Cattle",
        multiplier="",
        delayed=False,
    ),
    "GSCI": FutureSymbol(
        symbol="GSCI",
        exchange="GLOBEX",
        name="S&P-GSCI Index",
        multiplier="",
        delayed=False,
    ),
    "HUF": FutureSymbol(
        symbol="HUF",
        exchange="GLOBEX",
        name="Hungarian forint",
        multiplier="",
        delayed=False,
    ),
    "JPY": FutureSymbol(
        symbol="JPY",
        exchange="GLOBEX",
        name="Japanese yen",
        multiplier="",
        delayed=False,
    ),
    "LE": FutureSymbol(
        symbol="LE", exchange="GLOBEX", name="Live Cattle", multiplier="", delayed=False
    ),
    "HE": FutureSymbol(
        symbol="HE", exchange="GLOBEX", name="Lean Hogs", multiplier="", delayed=False
    ),
    "MXP": FutureSymbol(
        symbol="MXP",
        exchange="GLOBEX",
        name="Mexican Peso",
        multiplier="",
        delayed=False,
    ),
    "NZD": FutureSymbol(
        symbol="NZD",
        exchange="GLOBEX",
        name="New Zealand dollar",
        multiplier="",
        delayed=False,
    ),
    "NKD": FutureSymbol(
        symbol="NKD", exchange="GLOBEX", name="NIKKEI 225", multiplier="", delayed=False
    ),
    "PLN": FutureSymbol(
        symbol="PLN",
        exchange="GLOBEX",
        name="Polish zloty",
        multiplier="",
        delayed=False,
    ),
    "ZAR": FutureSymbol(
        symbol="ZAR",
        exchange="GLOBEX",
        name="South African Rand",
        multiplier="",
        delayed=False,
    ),
    "RF": FutureSymbol(
        symbol="RF",
        exchange="GLOBEX",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "RP": FutureSymbol(
        symbol="RP",
        exchange="GLOBEX",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "RUR": FutureSymbol(
        symbol="RUR",
        exchange="GLOBEX",
        name="Russian Ruble in US Dollars",
        multiplier="",
        delayed=False,
    ),
    "RY": FutureSymbol(
        symbol="RY",
        exchange="GLOBEX",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "CHF": FutureSymbol(
        symbol="CHF",
        exchange="GLOBEX",
        name="Swiss franc",
        multiplier="",
        delayed=False,
    ),
    "BRR": FutureSymbol(
        symbol="BRR",
        exchange="CMECRYPTO",
        name="CME CF Bitcoin Reference Rate",
        multiplier="",
        delayed=False,
    ),
    "ETHUSDRR": FutureSymbol(
        symbol="ETHUSDRR",
        exchange="CMECRYPTO",
        name="CME CF Ether-Dollar Reference Rate",
        multiplier="",
        delayed=False,
    ),
    "MBT": FutureSymbol(
        symbol="MBT",
        exchange="CMECRYPTO",
        name="Micro Bitcoin",
        multiplier="",
        delayed=False,
    ),
    "10Y": FutureSymbol(
        symbol="10Y",
        exchange="ECBOT",
        name="10 Year Micro Treasury Yield",
        multiplier="",
        delayed=False,
    ),
    "2YY": FutureSymbol(
        symbol="2YY",
        exchange="ECBOT",
        name="2 Year Micro Treasury Yield",
        multiplier="",
        delayed=False,
    ),
    "30Y": FutureSymbol(
        symbol="30Y",
        exchange="ECBOT",
        name="30 Year Micro Treasury Yield",
        multiplier="",
        delayed=False,
    ),
    "5YY": FutureSymbol(
        symbol="5YY",
        exchange="ECBOT",
        name="5 Year Micro Treasury Yield",
        multiplier="",
        delayed=False,
    ),
    "AIGCI": FutureSymbol(
        symbol="AIGCI",
        exchange="ECBOT",
        name="Bloomberg Commodity Index",
        multiplier="",
        delayed=False,
    ),
    "B1U": FutureSymbol(
        symbol="B1U",
        exchange="ECBOT",
        name="30-Year Deliverable Interest Rate Swap Futures",
        multiplier="",
        delayed=False,
    ),
    "AC": FutureSymbol(
        symbol="AC", exchange="ECBOT", name="Ethanol -CME", multiplier="", delayed=False
    ),
    "F1U": FutureSymbol(
        symbol="F1U",
        exchange="ECBOT",
        name="5-Year Deliverable Interest Rate Swap Futures",
        multiplier="",
        delayed=False,
    ),
    "KE": FutureSymbol(
        symbol="KE",
        exchange="ECBOT",
        name="Hard Red Winter Wheat -KCBOT-",
        multiplier="",
        delayed=False,
    ),
    "LIT": FutureSymbol(
        symbol="LIT",
        exchange="ECBOT",
        name="2-Year Eris Swap Futures",
        multiplier="",
        delayed=False,
    ),
    "LIW": FutureSymbol(
        symbol="LIW",
        exchange="ECBOT",
        name="5-Year Eris Swap Futures",
        multiplier="",
        delayed=False,
    ),
    "MYM": FutureSymbol(
        symbol="MYM",
        exchange="ECBOT",
        name="Micro E-mini DJIA",
        multiplier="",
        delayed=False,
    ),
    "N1U": FutureSymbol(
        symbol="N1U",
        exchange="ECBOT",
        name="10-Year Deliverable Interest Rate Swap Futures",
        multiplier="",
        delayed=False,
    ),
    "DJUSRE": FutureSymbol(
        symbol="DJUSRE",
        exchange="ECBOT",
        name="Dow Jones US Real Estate Index",
        multiplier="",
        delayed=False,
    ),
    "T1U": FutureSymbol(
        symbol="T1U",
        exchange="ECBOT",
        name="2-Year Deliverable Interest Rate Swap Futures",
        multiplier="",
        delayed=False,
    ),
    "TN": FutureSymbol(
        symbol="TN",
        exchange="ECBOT",
        name="10-YR T-NOTES",
        multiplier="",
        delayed=False,
    ),
    "UB": FutureSymbol(
        symbol="UB",
        exchange="ECBOT",
        name="Ultra T-BONDS",
        multiplier="",
        delayed=False,
    ),
    "YC": FutureSymbol(
        symbol="YC",
        exchange="ECBOT",
        name="Mini Sized Corn Futures",
        multiplier="",
        delayed=False,
    ),
    "YK": FutureSymbol(
        symbol="YK",
        exchange="ECBOT",
        name="Mini Sized Soybean Futures",
        multiplier="",
        delayed=False,
    ),
    "YW": FutureSymbol(
        symbol="YW",
        exchange="ECBOT",
        name="Mini Sized Wheat Futures",
        multiplier="",
        delayed=False,
    ),
    "YM": FutureSymbol(
        symbol="YM", exchange="ECBOT", name="MINI DJIA", multiplier="", delayed=False
    ),
    "Z3N": FutureSymbol(
        symbol="Z3N", exchange="ECBOT", name="3-YR TREAS.", multiplier="", delayed=False
    ),
    "ZB": FutureSymbol(
        symbol="ZB",
        exchange="ECBOT",
        name="30-year T-BONDS",
        multiplier="",
        delayed=False,
    ),
    "ZC": FutureSymbol(
        symbol="ZC", exchange="ECBOT", name="Corn Futures", multiplier="", delayed=False
    ),
    "ZF": FutureSymbol(
        symbol="ZF", exchange="ECBOT", name="5-YR TREAS.", multiplier="", delayed=False
    ),
    "ZL": FutureSymbol(
        symbol="ZL",
        exchange="ECBOT",
        name="Soybean Oil Futures",
        multiplier="",
        delayed=False,
    ),
    "ZM": FutureSymbol(
        symbol="ZM",
        exchange="ECBOT",
        name="Soybean Meal Futures",
        multiplier="",
        delayed=False,
    ),
    "ZN": FutureSymbol(
        symbol="ZN", exchange="ECBOT", name="10-YR TREAS.", multiplier="", delayed=False
    ),
    "ZO": FutureSymbol(
        symbol="ZO", exchange="ECBOT", name="Oat Futures", multiplier="", delayed=False
    ),
    "ZQ": FutureSymbol(
        symbol="ZQ",
        exchange="ECBOT",
        name="30 Day Federal Funds",
        multiplier="",
        delayed=False,
    ),
    "ZR": FutureSymbol(
        symbol="ZR",
        exchange="ECBOT",
        name="Rough Rice Futures",
        multiplier="",
        delayed=False,
    ),
    "ZS": FutureSymbol(
        symbol="ZS",
        exchange="ECBOT",
        name="Soybean Futures",
        multiplier="",
        delayed=False,
    ),
    "ZT": FutureSymbol(
        symbol="ZT", exchange="ECBOT", name="2-YR TREAS.", multiplier="", delayed=False
    ),
    "ZW": FutureSymbol(
        symbol="ZW",
        exchange="ECBOT",
        name="Wheat Futures",
        multiplier="",
        delayed=False,
    ),
    "BQX": FutureSymbol(
        symbol="BQX",
        exchange="GLOBEX",
        name="CME E-Mini NASDAQ Biotechnology",
        multiplier="",
        delayed=False,
    ),
    "BOS": FutureSymbol(
        symbol="BOS",
        exchange="GLOBEX",
        name="Boston Housing Index",
        multiplier="",
        delayed=False,
    ),
    "CB": FutureSymbol(
        symbol="CB",
        exchange="GLOBEX",
        name="CME Cash-Settled Butter Futures",
        multiplier="",
        delayed=False,
    ),
    "CHI": FutureSymbol(
        symbol="CHI",
        exchange="GLOBEX",
        name="Chicago Housing Index",
        multiplier="",
        delayed=False,
    ),
    "CLP": FutureSymbol(
        symbol="CLP",
        exchange="GLOBEX",
        name="Chilean peso",
        multiplier="",
        delayed=False,
    ),
    "CJY": FutureSymbol(
        symbol="CJY",
        exchange="GLOBEX",
        name="Canadian dollar",
        multiplier="",
        delayed=False,
    ),
    "CNH": FutureSymbol(
        symbol="CNH",
        exchange="GLOBEX",
        name="United States dollar",
        multiplier="",
        delayed=False,
    ),
    "CSC": FutureSymbol(
        symbol="CSC", exchange="GLOBEX", name="Cheese", multiplier="", delayed=False
    ),
    "CUS": FutureSymbol(
        symbol="CUS",
        exchange="GLOBEX",
        name="Housing Index Composite",
        multiplier="",
        delayed=False,
    ),
    "DA": FutureSymbol(
        symbol="DA",
        exchange="GLOBEX",
        name="MILK CLASS III INDEX",
        multiplier="",
        delayed=False,
    ),
    "DEN": FutureSymbol(
        symbol="DEN",
        exchange="GLOBEX",
        name="Denver Housing Index",
        multiplier="",
        delayed=False,
    ),
    "DY": FutureSymbol(
        symbol="DY",
        exchange="GLOBEX",
        name="CME DRY WHEY INDEX",
        multiplier="",
        delayed=False,
    ),
    "E7": FutureSymbol(
        symbol="E7",
        exchange="GLOBEX",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "EAD": FutureSymbol(
        symbol="EAD",
        exchange="GLOBEX",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "ECD": FutureSymbol(
        symbol="ECD",
        exchange="GLOBEX",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "EMD": FutureSymbol(
        symbol="EMD",
        exchange="GLOBEX",
        name="E-mini S&P Midcap 400 Futures",
        multiplier="",
        delayed=False,
    ),
    "NIY": FutureSymbol(
        symbol="NIY",
        exchange="GLOBEX",
        name="Yen Denominated Nikkei 225 Index",
        multiplier="",
        delayed=False,
    ),
    "ES": FutureSymbol(
        symbol="ES",
        exchange="GLOBEX",
        name="MINI-S&P 500",
        multiplier="",
        delayed=False,
    ),
    "SPXESUP": FutureSymbol(
        symbol="SPXESUP",
        exchange="GLOBEX",
        name="E-mini S&P 500 ESG",
        multiplier="",
        delayed=False,
    ),
    "GDK": FutureSymbol(
        symbol="GDK",
        exchange="GLOBEX",
        name="Class IV Milk - 200k lbs",
        multiplier="",
        delayed=False,
    ),
    "NF": FutureSymbol(
        symbol="NF",
        exchange="GLOBEX",
        name="NON FAT DRY MILK INDEX",
        multiplier="",
        delayed=False,
    ),
    "IBAA": FutureSymbol(
        symbol="IBAA",
        exchange="GLOBEX",
        name="Bovespa Index - USD",
        multiplier="",
        delayed=False,
    ),
    "ILS": FutureSymbol(
        symbol="ILS",
        exchange="GLOBEX",
        name="Israeli Shekel in US Dollar",
        multiplier="",
        delayed=False,
    ),
    "J7": FutureSymbol(
        symbol="J7",
        exchange="GLOBEX",
        name="Japanese yen",
        multiplier="",
        delayed=False,
    ),
    "KRW": FutureSymbol(
        symbol="KRW", exchange="GLOBEX", name="Korean Won", multiplier="", delayed=False
    ),
    "LAV": FutureSymbol(
        symbol="LAV",
        exchange="GLOBEX",
        name="Las Vegas Housing Index",
        multiplier="",
        delayed=False,
    ),
    "LAX": FutureSymbol(
        symbol="LAX",
        exchange="GLOBEX",
        name="Los Angeles Housing Index",
        multiplier="",
        delayed=False,
    ),
    "LB": FutureSymbol(
        symbol="LB",
        exchange="GLOBEX",
        name="Random Length Lumber",
        multiplier="",
        delayed=False,
    ),
    "M2K": FutureSymbol(
        symbol="M2K",
        exchange="GLOBEX",
        name="Micro E-mini Russell 2000",
        multiplier="",
        delayed=False,
    ),
    "M6A": FutureSymbol(
        symbol="M6A",
        exchange="GLOBEX",
        name="Australian dollar",
        multiplier="",
        delayed=False,
    ),
    "M6B": FutureSymbol(
        symbol="M6B",
        exchange="GLOBEX",
        name="British pound",
        multiplier="",
        delayed=False,
    ),
    "M6C": FutureSymbol(
        symbol="M6C",
        exchange="GLOBEX",
        name="United States dollar",
        multiplier="",
        delayed=False,
    ),
    "M6E": FutureSymbol(
        symbol="M6E",
        exchange="GLOBEX",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "M6J": FutureSymbol(
        symbol="M6J",
        exchange="GLOBEX",
        name="United States dollar",
        multiplier="",
        delayed=False,
    ),
    "M6S": FutureSymbol(
        symbol="M6S",
        exchange="GLOBEX",
        name="United States dollar",
        multiplier="",
        delayed=False,
    ),
    "MCD": FutureSymbol(
        symbol="MCD",
        exchange="GLOBEX",
        name="Canadian dollar",
        multiplier="",
        delayed=False,
    ),
    "MES": FutureSymbol(
        symbol="MES",
        exchange="GLOBEX",
        name="Micro E-mini S&P 500",
        multiplier="",
        delayed=False,
    ),
    "MIA": FutureSymbol(
        symbol="MIA",
        exchange="GLOBEX",
        name="Miami Housing Index",
        multiplier="",
        delayed=False,
    ),
    "MIR": FutureSymbol(
        symbol="MIR",
        exchange="GLOBEX",
        name="Indian Rupee",
        multiplier="",
        delayed=False,
    ),
    "MJY": FutureSymbol(
        symbol="MJY",
        exchange="GLOBEX",
        name="Japanese yen",
        multiplier="",
        delayed=False,
    ),
    "MNH": FutureSymbol(
        symbol="MNH",
        exchange="GLOBEX",
        name="United States dollar",
        multiplier="",
        delayed=False,
    ),
    "MNQ": FutureSymbol(
        symbol="MNQ",
        exchange="GLOBEX",
        name="Micro E-mini NASDAQ-100",
        multiplier="",
        delayed=False,
    ),
    "MSF": FutureSymbol(
        symbol="MSF",
        exchange="GLOBEX",
        name="Swiss franc",
        multiplier="",
        delayed=False,
    ),
    "NOK": FutureSymbol(
        symbol="NOK",
        exchange="GLOBEX",
        name="Norwegian krone",
        multiplier="",
        delayed=False,
    ),
    "NQ": FutureSymbol(
        symbol="NQ",
        exchange="GLOBEX",
        name="NASDAQ E-MINI",
        multiplier="",
        delayed=False,
    ),
    "NYM": FutureSymbol(
        symbol="NYM",
        exchange="GLOBEX",
        name="New York Housing Index",
        multiplier="",
        delayed=False,
    ),
    "PJY": FutureSymbol(
        symbol="PJY",
        exchange="GLOBEX",
        name="British pound",
        multiplier="",
        delayed=False,
    ),
    "PSF": FutureSymbol(
        symbol="PSF",
        exchange="GLOBEX",
        name="British pound",
        multiplier="",
        delayed=False,
    ),
    "RMB": FutureSymbol(
        symbol="RMB",
        exchange="GLOBEX",
        name="CME Chinese Renminbi in US Dollar Cross Rate",
        multiplier="",
        delayed=False,
    ),
    "RME": FutureSymbol(
        symbol="RME",
        exchange="GLOBEX",
        name="CME Chinese Renminbi in Euro Cross Rate",
        multiplier="",
        delayed=False,
    ),
    "RS1": FutureSymbol(
        symbol="RS1",
        exchange="GLOBEX",
        name="E-mini Russell 1000 Index Futures",
        multiplier="",
        delayed=False,
    ),
    "RSG": FutureSymbol(
        symbol="RSG",
        exchange="GLOBEX",
        name="E-mini Russell 1000 Growth Index Futures",
        multiplier="",
        delayed=False,
    ),
    "RSV": FutureSymbol(
        symbol="RSV",
        exchange="GLOBEX",
        name="E-Mini Russell 1000 Value Index Futures",
        multiplier="",
        delayed=False,
    ),
    "RTY": FutureSymbol(
        symbol="RTY",
        exchange="GLOBEX",
        name="E-Mini Russell 2000 Index",
        multiplier="",
        delayed=False,
    ),
    "SPXDIVAN": FutureSymbol(
        symbol="SPXDIVAN",
        exchange="GLOBEX",
        name="S&P 500 Dividend Points (Annual) Index",
        multiplier="",
        delayed=False,
    ),
    "SDG": FutureSymbol(
        symbol="SDG",
        exchange="GLOBEX",
        name="San Diego Housing Index",
        multiplier="",
        delayed=False,
    ),
    "SEK": FutureSymbol(
        symbol="SEK",
        exchange="GLOBEX",
        name="Swedish krona",
        multiplier="",
        delayed=False,
    ),
    "SFR": FutureSymbol(
        symbol="SFR",
        exchange="GLOBEX",
        name="San Francisco Housing Index",
        multiplier="",
        delayed=False,
    ),
    "SGX": FutureSymbol(
        symbol="SGX",
        exchange="GLOBEX",
        name="S&P 500 / Citigroup Growth Index",
        multiplier="",
        delayed=False,
    ),
    "SIR": FutureSymbol(
        symbol="SIR",
        exchange="GLOBEX",
        name="Indian Rupee",
        multiplier="",
        delayed=False,
    ),
    "SJY": FutureSymbol(
        symbol="SJY",
        exchange="GLOBEX",
        name="Swiss franc",
        multiplier="",
        delayed=False,
    ),
    "SMC": FutureSymbol(
        symbol="SMC",
        exchange="GLOBEX",
        name="E-Mini S&P SmallCap 600 Futures",
        multiplier="",
        delayed=False,
    ),
    "SONIA": FutureSymbol(
        symbol="SONIA",
        exchange="GLOBEX",
        name="Sterling Overnight Index Average",
        multiplier="",
        delayed=False,
    ),
    "SOFR1": FutureSymbol(
        symbol="SOFR1",
        exchange="GLOBEX",
        name="Secured Overnight Financing Rate 1-month average of rates",
        multiplier="",
        delayed=False,
    ),
    "SOFR3": FutureSymbol(
        symbol="SOFR3",
        exchange="GLOBEX",
        name="Secured Overnight Financing Rate 3-month average of rates",
        multiplier="",
        delayed=False,
    ),
    "SVX": FutureSymbol(
        symbol="SVX",
        exchange="GLOBEX",
        name="S&P 500 / Citigroup Value Index",
        multiplier="",
        delayed=False,
    ),
    "WDC": FutureSymbol(
        symbol="WDC",
        exchange="GLOBEX",
        name="Washington DC Housing Index",
        multiplier="",
        delayed=False,
    ),
    "IXB": FutureSymbol(
        symbol="IXB",
        exchange="GLOBEX",
        name="Materials Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXE": FutureSymbol(
        symbol="IXE",
        exchange="GLOBEX",
        name="Energy Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXM": FutureSymbol(
        symbol="IXM",
        exchange="GLOBEX",
        name="Financial Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXI": FutureSymbol(
        symbol="IXI",
        exchange="GLOBEX",
        name="Industrial Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXT": FutureSymbol(
        symbol="IXT",
        exchange="GLOBEX",
        name="Technology Select Sector Index -",
        multiplier="",
        delayed=False,
    ),
    "IXR": FutureSymbol(
        symbol="IXR",
        exchange="GLOBEX",
        name="Consumer Staples Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXRE": FutureSymbol(
        symbol="IXRE",
        exchange="GLOBEX",
        name="Real Estate Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXU": FutureSymbol(
        symbol="IXU",
        exchange="GLOBEX",
        name="Utilities Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXV": FutureSymbol(
        symbol="IXV",
        exchange="GLOBEX",
        name="Health Care Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXY": FutureSymbol(
        symbol="IXY",
        exchange="GLOBEX",
        name="Consumer Discretionary Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "BAKKT": FutureSymbol(
        symbol="BAKKT",
        exchange="ICECRYPTO",
        name="Bakkt Bitcoin",
        multiplier="",
        delayed=False,
    ),
    "CC": FutureSymbol(
        symbol="CC", exchange="NYBOT", name="Cocoa NYBOT", multiplier="", delayed=False
    ),
    "CT": FutureSymbol(
        symbol="CT", exchange="NYBOT", name="Cotton No. 2", multiplier="", delayed=False
    ),
    "DX": FutureSymbol(
        symbol="DX",
        exchange="NYBOT",
        name="NYBOT US Dollar FX",
        multiplier="",
        delayed=False,
    ),
    "NYFANG": FutureSymbol(
        symbol="NYFANG",
        exchange="NYBOT",
        name="NYSE FANG+ Index",
        multiplier="",
        delayed=False,
    ),
    "KC": FutureSymbol(
        symbol="KC", exchange="NYBOT", name='Coffee "C"', multiplier="", delayed=False
    ),
    "OJ": FutureSymbol(
        symbol="OJ",
        exchange="NYBOT",
        name='FC Orange Juice "A"',
        multiplier="",
        delayed=False,
    ),
    "RS": FutureSymbol(
        symbol="RS", exchange="NYBOT", name="Canola", multiplier="", delayed=False
    ),
    "SB": FutureSymbol(
        symbol="SB", exchange="NYBOT", name="Sugar No. 11", multiplier="", delayed=False
    ),
    "SF": FutureSymbol(
        symbol="SF",
        exchange="NYBOT",
        name="Sugar #16 112000 lbs",
        multiplier="",
        delayed=False,
    ),
    "ALI": FutureSymbol(
        symbol="ALI",
        exchange="NYMEX",
        name="NYMEX Aluminum Index",
        multiplier="",
        delayed=False,
    ),
    "BB": FutureSymbol(
        symbol="BB",
        exchange="NYMEX",
        name="NYMEX Brent Financial Futures Index",
        multiplier="",
        delayed=False,
    ),
    "BZ": FutureSymbol(
        symbol="BZ",
        exchange="NYMEX",
        name="Brent Crude Oil - Last Day",
        multiplier="",
        delayed=False,
    ),
    "CL": FutureSymbol(
        symbol="CL", exchange="NYMEX", name="Crude oil", multiplier="", delayed=False
    ),
    "GC": FutureSymbol(
        symbol="GC", exchange="NYMEX", name="Gold", multiplier="", delayed=False
    ),
    "HG": FutureSymbol(
        symbol="HG", exchange="NYMEX", name="Copper", multiplier="", delayed=False
    ),
    "HH": FutureSymbol(
        symbol="HH",
        exchange="NYMEX",
        name="Nautral Gas Last Day Financial  Future",
        multiplier="",
        delayed=False,
    ),
    "HO": FutureSymbol(
        symbol="HO", exchange="NYMEX", name="Heating Oil", multiplier="", delayed=False
    ),
    "HP": FutureSymbol(
        symbol="HP",
        exchange="NYMEX",
        name="Natural Gas Penultimate Financial Futures Index",
        multiplier="",
        delayed=False,
    ),
    "HRC": FutureSymbol(
        symbol="HRC",
        exchange="NYMEX",
        name="Hot-Rolled Coil Steel Index",
        multiplier="",
        delayed=False,
    ),
    "LT": FutureSymbol(
        symbol="LT",
        exchange="NYMEX",
        name="Gulf Coast ULSD (Platts) Up-Down Futures",
        multiplier="",
        delayed=False,
    ),
    "MCL": FutureSymbol(
        symbol="MCL",
        exchange="NYMEX",
        name="Micro WTI Crude Oil",
        multiplier="",
        delayed=False,
    ),
    "MGC": FutureSymbol(
        symbol="MGC",
        exchange="NYMEX",
        name="E-Micro Gold",
        multiplier="",
        delayed=False,
    ),
    "NG": FutureSymbol(
        symbol="NG", exchange="NYMEX", name="Natural gas", multiplier="", delayed=False
    ),
    "PA": FutureSymbol(
        symbol="PA",
        exchange="NYMEX",
        name="NYMEX Palladium Index",
        multiplier="",
        delayed=False,
    ),
    "PL": FutureSymbol(
        symbol="PL",
        exchange="NYMEX",
        name="NYMEX Platinum Index",
        multiplier="",
        delayed=False,
    ),
    "QC": FutureSymbol(
        symbol="QC", exchange="NYMEX", name="Copper", multiplier="", delayed=False
    ),
    "QG": FutureSymbol(
        symbol="QG",
        exchange="NYMEX",
        name="Natural gas E-Mini",
        multiplier="",
        delayed=False,
    ),
    "QI": FutureSymbol(
        symbol="QI", exchange="NYMEX", name="Silver Mini", multiplier="", delayed=False
    ),
    "QM": FutureSymbol(
        symbol="QM",
        exchange="NYMEX",
        name="Crude oil E-Mini",
        multiplier="",
        delayed=False,
    ),
    "QO": FutureSymbol(
        symbol="QO", exchange="NYMEX", name="Gold", multiplier="", delayed=False
    ),
    "RB": FutureSymbol(
        symbol="RB",
        exchange="NYMEX",
        name="RBOB Gasoline",
        multiplier="",
        delayed=False,
    ),
    "SGC": FutureSymbol(
        symbol="SGC",
        exchange="NYMEX",
        name="Shanghai Gold Exchange Gold Benchmark PM Price Index - CNH Futures",
        multiplier="",
        delayed=False,
    ),
    "SGUF": FutureSymbol(
        symbol="SGUF",
        exchange="NYMEX",
        name="Shanghai Gold Exchange Gold Benchmark PM Price Index - USD Futures",
        multiplier="",
        delayed=False,
    ),
    "SI": FutureSymbol(
        symbol="SI", exchange="NYMEX", name="Silver", multiplier="", delayed=False
    ),
    "TT": FutureSymbol(
        symbol="TT",
        exchange="NYMEX",
        name="NYMEX Cotton index",
        multiplier="",
        delayed=False,
    ),
    "UX": FutureSymbol(
        symbol="UX",
        exchange="NYMEX",
        name="NYMEX Uranium Index",
        multiplier="",
        delayed=False,
    ),
    "10YSME": FutureSymbol(
        symbol="10YSME",
        exchange="SMFE",
        name="Small 10YR US Treasury Yield",
        multiplier="",
        delayed=False,
    ),
    "2YSME": FutureSymbol(
        symbol="2YSME",
        exchange="SMFE",
        name="Small 2YR US Treasury Yield",
        multiplier="",
        delayed=False,
    ),
    "30YSME": FutureSymbol(
        symbol="30YSME",
        exchange="SMFE",
        name="Small 30YR US Treasury Yield",
        multiplier="",
        delayed=False,
    ),
    "S420": FutureSymbol(
        symbol="S420",
        exchange="SMFE",
        name="Small Cannabis",
        multiplier="",
        delayed=False,
    ),
    "SCCX": FutureSymbol(
        symbol="SCCX",
        exchange="SMFE",
        name="Small Cryptocurrency",
        multiplier="",
        delayed=False,
    ),
    "FXSME": FutureSymbol(
        symbol="FXSME",
        exchange="SMFE",
        name="Small US Dollar",
        multiplier="",
        delayed=False,
    ),
    "75SME": FutureSymbol(
        symbol="75SME",
        exchange="SMFE",
        name="Small Stocks 75",
        multiplier="",
        delayed=False,
    ),
    "SMO": FutureSymbol(
        symbol="SMO",
        exchange="SMFE",
        name="Small US Crude Oil",
        multiplier="",
        delayed=False,
    ),
    "PRESME": FutureSymbol(
        symbol="PRESME",
        exchange="SMFE",
        name="Small Precious Metals",
        multiplier="",
        delayed=False,
    ),
    "STXSME": FutureSymbol(
        symbol="STXSME",
        exchange="SMFE",
        name="Small Technology 60",
        multiplier="",
        delayed=False,
    ),
    "SP": FutureSymbol(
        symbol="SP", exchange="GLOBEX", name="S&P 500", multiplier="", delayed=False
    ),
    "QH": FutureSymbol(
        symbol="QH",
        exchange="NYMEX",
        name="Heating Oil E-Mini",
        multiplier="",
        delayed=False,
    ),
    "QU": FutureSymbol(
        symbol="QU",
        exchange="NYMEX",
        name="Unleaded Gasoline E-Mini",
        multiplier="",
        delayed=False,
    ),
    "SIL": FutureSymbol(
        symbol="SIL", exchange="NYMEX", name="Silver", multiplier="", delayed=False
    ),
    "TF": FutureSymbol(
        symbol="TF", exchange="NYBOT", name="RUSSELL 2000", multiplier="", delayed=False
    ),
    # NOTABLE MANUAL EXCEPTIONS TO THE ABOVE:
    # IBKR uses the SAME SYMBOL for bitcoin futures and micro bitcoin futures, with
    # the only difference being the multiplier requirement.
    # We distinguish our usable names via /BTC for full and /MBT for micros.
    "BTC": FutureSymbol(
        symbol="BRR",
        exchange="CMECRYPTO",
        multiplier=5,
        name="CME CF Bitcoin Reference Rate",
    ),
    "MBT": FutureSymbol(
        symbol="BRR",
        exchange="CMECRYPTO",
        multiplier=0.1,
        name="CME CF Micro Bitcoin Reference Rate",
    ),
    # We can't figure out which data package enables VIX access, so use delayed quotes.
    "VIX": FutureSymbol(symbol="VIX", exchange="CFE", name="CBOE Volatility Index"),
    "VXM": FutureSymbol(
        symbol="VXM",
        delayed=True,
        exchange="CFE",
        name="Mini Cboe Volatility Index Futures",
    ),
}
