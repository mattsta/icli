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
                "CBOT",
                "CME",
                "GLOBEX",
                "CMECRYPTO",
                "CFE",
                "NYBOT",
                "NYMEX",
                "SMFE",
                "ICECRYPTO",
            }:
                # why setdefault()? Because we only want the FIRST TIME we see a symbol to generate
                # an entry here. This page as additional tables after the ones we want and we DO NOT WANT
                # the final "time at exchange" tables to pollute our previously-populated correct values.
                fs.setdefault(
                    row[1], FutureSymbol(symbol=symbol, exchange=exchange, name=name)
                )

    return fs


# simple mapping from name of future to exchange for future.
# Used for generating ibkr api Future Contract specification since
# each symbol must have an exchange declared.
FUTS_EXCHANGE = {
    "10Y": FutureSymbol(
        symbol="10Y",
        exchange="CBOT",
        name="10 Year Micro Treasury Yield",
        multiplier="",
        delayed=False,
    ),
    "2YY": FutureSymbol(
        symbol="2YY",
        exchange="CBOT",
        name="2 Year Micro Treasury Yield",
        multiplier="",
        delayed=False,
    ),
    "30Y": FutureSymbol(
        symbol="30Y",
        exchange="CBOT",
        name="30 Year Micro Treasury Yield",
        multiplier="",
        delayed=False,
    ),
    "5YY": FutureSymbol(
        symbol="5YY",
        exchange="CBOT",
        name="5 Year Micro Treasury Yield",
        multiplier="",
        delayed=False,
    ),
    "AIGCI": FutureSymbol(
        symbol="AIGCI",
        exchange="CBOT",
        name="Bloomberg Commodity Index",
        multiplier="",
        delayed=False,
    ),
    "AC": FutureSymbol(
        symbol="AC", exchange="CBOT", name="Ethanol -CME", multiplier="", delayed=False
    ),
    "KE": FutureSymbol(
        symbol="KE",
        exchange="CBOT",
        name="Hard Red Winter Wheat -KCBOT-",
        multiplier="",
        delayed=False,
    ),
    "MTN": FutureSymbol(
        symbol="MTN",
        exchange="CBOT",
        name="Micro Ultra 10-Year U.S Treasury Note",
        multiplier="",
        delayed=False,
    ),
    "MWN": FutureSymbol(
        symbol="MWN",
        exchange="CBOT",
        name="Micro Ultra US Treasury Bond",
        multiplier="",
        delayed=False,
    ),
    "MYM": FutureSymbol(
        symbol="MYM",
        exchange="CBOT",
        name="Micro E-Mini Dow Jones Industrial Average Index",
        multiplier="",
        delayed=False,
    ),
    "DJUSRE": FutureSymbol(
        symbol="DJUSRE",
        exchange="CBOT",
        name="Dow Jones US Real Estate Index",
        multiplier="",
        delayed=False,
    ),
    "TN": FutureSymbol(
        symbol="TN",
        exchange="CBOT",
        name="Ultra 10-Year US Treasury Note",
        multiplier="",
        delayed=False,
    ),
    "TWE": FutureSymbol(
        symbol="TWE",
        exchange="CBOT",
        name="20-Year U.S. Treasury Bond",
        multiplier="",
        delayed=False,
    ),
    "UB": FutureSymbol(
        symbol="UB",
        exchange="CBOT",
        name="Ultra Treasury Bond",
        multiplier="",
        delayed=False,
    ),
    "YC": FutureSymbol(
        symbol="YC",
        exchange="CBOT",
        name="Mini Sized Corn Futures",
        multiplier="",
        delayed=False,
    ),
    "YK": FutureSymbol(
        symbol="YK",
        exchange="CBOT",
        name="Mini Sized Soybean Futures",
        multiplier="",
        delayed=False,
    ),
    "YW": FutureSymbol(
        symbol="YW",
        exchange="CBOT",
        name="Mini Sized Wheat Futures",
        multiplier="",
        delayed=False,
    ),
    "YIA": FutureSymbol(
        symbol="YIA",
        exchange="CBOT",
        name="1-Year Eris SOFR Swap Futures",
        multiplier="",
        delayed=False,
    ),
    "YIC": FutureSymbol(
        symbol="YIC",
        exchange="CBOT",
        name="3-Year Eris SOFR Swap Futures",
        multiplier="",
        delayed=False,
    ),
    "YID": FutureSymbol(
        symbol="YID",
        exchange="CBOT",
        name="4-Year Eris SOFR Swap Futures",
        multiplier="",
        delayed=False,
    ),
    "YIT": FutureSymbol(
        symbol="YIT",
        exchange="CBOT",
        name="2-Year Eris SOFR Swap Futures",
        multiplier="",
        delayed=False,
    ),
    "YIW": FutureSymbol(
        symbol="YIW",
        exchange="CBOT",
        name="5-Year Eris SOFR Swap Futures",
        multiplier="",
        delayed=False,
    ),
    "YM": FutureSymbol(
        symbol="YM",
        exchange="CBOT",
        name="E-mini Dow Jones Industrial Average",
        multiplier="",
        delayed=False,
    ),
    "Z3N": FutureSymbol(
        symbol="Z3N",
        exchange="CBOT",
        name="3 YEAR US TREASURY NOTE",
        multiplier="",
        delayed=False,
    ),
    "ZB": FutureSymbol(
        symbol="ZB",
        exchange="CBOT",
        name="US Treasury Bond",
        multiplier="",
        delayed=False,
    ),
    "ZC": FutureSymbol(
        symbol="ZC", exchange="CBOT", name="Corn Futures", multiplier="", delayed=False
    ),
    "ZF": FutureSymbol(
        symbol="ZF",
        exchange="CBOT",
        name="5 Year US Treasury Note",
        multiplier="",
        delayed=False,
    ),
    "ZL": FutureSymbol(
        symbol="ZL",
        exchange="CBOT",
        name="Soybean Oil Futures",
        multiplier="",
        delayed=False,
    ),
    "ZM": FutureSymbol(
        symbol="ZM",
        exchange="CBOT",
        name="Soybean Meal Futures",
        multiplier="",
        delayed=False,
    ),
    "ZN": FutureSymbol(
        symbol="ZN",
        exchange="CBOT",
        name="10 Year US Treasury Note",
        multiplier="",
        delayed=False,
    ),
    "ZO": FutureSymbol(
        symbol="ZO", exchange="CBOT", name="Oat Futures", multiplier="", delayed=False
    ),
    "ZQ": FutureSymbol(
        symbol="ZQ",
        exchange="CBOT",
        name="30 Day Fed Funds",
        multiplier="",
        delayed=False,
    ),
    "ZR": FutureSymbol(
        symbol="ZR",
        exchange="CBOT",
        name="Rough Rice Futures",
        multiplier="",
        delayed=False,
    ),
    "ZS": FutureSymbol(
        symbol="ZS",
        exchange="CBOT",
        name="Soybean Futures",
        multiplier="",
        delayed=False,
    ),
    "ZT": FutureSymbol(
        symbol="ZT",
        exchange="CBOT",
        name="2 Year US Treasury Note",
        multiplier="",
        delayed=False,
    ),
    "ZW": FutureSymbol(
        symbol="ZW", exchange="CBOT", name="Wheat Futures", multiplier="", delayed=False
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
    "AUD": FutureSymbol(
        symbol="AUD",
        exchange="CME",
        name="Australian dollar",
        multiplier="",
        delayed=False,
    ),
    "GBP": FutureSymbol(
        symbol="GBP", exchange="CME", name="British pound", multiplier="", delayed=False
    ),
    "CAD": FutureSymbol(
        symbol="CAD",
        exchange="CME",
        name="Canadian dollar",
        multiplier="",
        delayed=False,
    ),
    "EUR": FutureSymbol(
        symbol="EUR",
        exchange="CME",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "JPY": FutureSymbol(
        symbol="JPY", exchange="CME", name="Japanese yen", multiplier="", delayed=False
    ),
    "BRE": FutureSymbol(
        symbol="BRE",
        exchange="CME",
        name="Brazilian real",
        multiplier="",
        delayed=False,
    ),
    "MXP": FutureSymbol(
        symbol="MXP", exchange="CME", name="Mexican Peso", multiplier="", delayed=False
    ),
    "NZD": FutureSymbol(
        symbol="NZD",
        exchange="CME",
        name="New Zealand dollar",
        multiplier="",
        delayed=False,
    ),
    "RUR *": FutureSymbol(
        symbol="RUR *",
        exchange="CME",
        name="Russian Ruble in US Dollars",
        multiplier="",
        delayed=False,
    ),
    "CHF": FutureSymbol(
        symbol="CHF", exchange="CME", name="Swiss franc", multiplier="", delayed=False
    ),
    "ZAR": FutureSymbol(
        symbol="ZAR",
        exchange="CME",
        name="South African Rand",
        multiplier="",
        delayed=False,
    ),
    "ACD": FutureSymbol(
        symbol="ACD",
        exchange="CME",
        name="Australian dollar",
        multiplier="",
        delayed=False,
    ),
    "AJY": FutureSymbol(
        symbol="AJY",
        exchange="CME",
        name="Australian dollar",
        multiplier="",
        delayed=False,
    ),
    "BQX": FutureSymbol(
        symbol="BQX",
        exchange="CME",
        name="CME E-Mini NASDAQ Biotechnology",
        multiplier="",
        delayed=False,
    ),
    "BOS": FutureSymbol(
        symbol="BOS",
        exchange="CME",
        name="Boston Housing Index",
        multiplier="",
        delayed=False,
    ),
    "BSBY": FutureSymbol(
        symbol="BSBY",
        exchange="CME",
        name="Three-Month Bloomberg Short-Term Bank Yield",
        multiplier="",
        delayed=False,
    ),
    "BRR": FutureSymbol(
        symbol="BRR",
        exchange="CME",
        name="CME CF Bitcoin Reference Rate",
        multiplier="",
        delayed=False,
    ),
    "BTCEURRR": FutureSymbol(
        symbol="BTCEURRR",
        exchange="CME",
        name="CME CF Bitcoin-Euro Reference Rate",
        multiplier="",
        delayed=False,
    ),
    "CB": FutureSymbol(
        symbol="CB",
        exchange="CME",
        name="CME Cash-Settled Butter Futures",
        multiplier="",
        delayed=False,
    ),
    "CHI": FutureSymbol(
        symbol="CHI",
        exchange="CME",
        name="Chicago Housing Index",
        multiplier="",
        delayed=False,
    ),
    "CLP": FutureSymbol(
        symbol="CLP", exchange="CME", name="Chilean peso", multiplier="", delayed=False
    ),
    "CJY": FutureSymbol(
        symbol="CJY",
        exchange="CME",
        name="Canadian dollar",
        multiplier="",
        delayed=False,
    ),
    "CNH": FutureSymbol(
        symbol="CNH",
        exchange="CME",
        name="United States dollar",
        multiplier="",
        delayed=False,
    ),
    "CSC": FutureSymbol(
        symbol="CSC", exchange="CME", name="Cheese", multiplier="", delayed=False
    ),
    "CUS": FutureSymbol(
        symbol="CUS",
        exchange="CME",
        name="Housing Index Composite",
        multiplier="",
        delayed=False,
    ),
    "CZK": FutureSymbol(
        symbol="CZK", exchange="CME", name="Czech koruna", multiplier="", delayed=False
    ),
    "DA": FutureSymbol(
        symbol="DA",
        exchange="CME",
        name="MILK CLASS III INDEX",
        multiplier="",
        delayed=False,
    ),
    "DEN": FutureSymbol(
        symbol="DEN",
        exchange="CME",
        name="Denver Housing Index",
        multiplier="",
        delayed=False,
    ),
    "DY": FutureSymbol(
        symbol="DY",
        exchange="CME",
        name="CME DRY WHEY INDEX",
        multiplier="",
        delayed=False,
    ),
    "E7": FutureSymbol(
        symbol="E7",
        exchange="CME",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "EAD": FutureSymbol(
        symbol="EAD",
        exchange="CME",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "ECD": FutureSymbol(
        symbol="ECD",
        exchange="CME",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "ECK": FutureSymbol(
        symbol="ECK", exchange="CME", name="Czech koruna", multiplier="", delayed=False
    ),
    "EHF": FutureSymbol(
        symbol="EHF",
        exchange="CME",
        name="Hungarian forint",
        multiplier="",
        delayed=False,
    ),
    "EMD": FutureSymbol(
        symbol="EMD",
        exchange="CME",
        name="E-mini S&P Midcap 400 Futures",
        multiplier="",
        delayed=False,
    ),
    "NIY": FutureSymbol(
        symbol="NIY",
        exchange="CME",
        name="Yen Denominated Nikkei 225 Index",
        multiplier="",
        delayed=False,
    ),
    "EPZ": FutureSymbol(
        symbol="EPZ", exchange="CME", name="Polish zloty", multiplier="", delayed=False
    ),
    "ES": FutureSymbol(
        symbol="ES", exchange="CME", name="E-mini S&P 500", multiplier="", delayed=False
    ),
    "SPXESUP": FutureSymbol(
        symbol="SPXESUP",
        exchange="CME",
        name="E-mini S&P 500 ESG",
        multiplier="",
        delayed=False,
    ),
    "ESTR": FutureSymbol(
        symbol="ESTR",
        exchange="CME",
        name="Euro Short-Term Rate",
        multiplier="",
        delayed=False,
    ),
    "ETHEURRR": FutureSymbol(
        symbol="ETHEURRR",
        exchange="CME",
        name="CME CF Ether-Euro Reference Rate",
        multiplier="",
        delayed=False,
    ),
    "ETHUSDRR": FutureSymbol(
        symbol="ETHUSDRR",
        exchange="CME",
        name="CME CF Ether-Dollar Reference Rate",
        multiplier="",
        delayed=False,
    ),
    "GSCI": FutureSymbol(
        symbol="GSCI",
        exchange="CME",
        name="S&P-GSCI Index",
        multiplier="",
        delayed=False,
    ),
    "GDK": FutureSymbol(
        symbol="GDK",
        exchange="CME",
        name="Class IV Milk - 200k lbs",
        multiplier="",
        delayed=False,
    ),
    "GF": FutureSymbol(
        symbol="GF", exchange="CME", name="Feeder Cattle", multiplier="", delayed=False
    ),
    "NF": FutureSymbol(
        symbol="NF",
        exchange="CME",
        name="NON FAT DRY MILK INDEX",
        multiplier="",
        delayed=False,
    ),
    "HE": FutureSymbol(
        symbol="HE", exchange="CME", name="Lean Hogs", multiplier="", delayed=False
    ),
    "HUF": FutureSymbol(
        symbol="HUF",
        exchange="CME",
        name="Hungarian forint",
        multiplier="",
        delayed=False,
    ),
    "IBAA": FutureSymbol(
        symbol="IBAA",
        exchange="CME",
        name="Bovespa Index - USD",
        multiplier="",
        delayed=False,
    ),
    "ILS": FutureSymbol(
        symbol="ILS",
        exchange="CME",
        name="Israeli Shekel",
        multiplier="",
        delayed=False,
    ),
    "J7": FutureSymbol(
        symbol="J7", exchange="CME", name="Japanese yen", multiplier="", delayed=False
    ),
    "KRW": FutureSymbol(
        symbol="KRW", exchange="CME", name="Korean Won", multiplier="", delayed=False
    ),
    "LAV": FutureSymbol(
        symbol="LAV",
        exchange="CME",
        name="Las Vegas Housing Index",
        multiplier="",
        delayed=False,
    ),
    "LAX": FutureSymbol(
        symbol="LAX",
        exchange="CME",
        name="Los Angeles Housing Index",
        multiplier="",
        delayed=False,
    ),
    "LBR": FutureSymbol(
        symbol="LBR",
        exchange="CME",
        name="Lumber Futures",
        multiplier="",
        delayed=False,
    ),
    "LE": FutureSymbol(
        symbol="LE", exchange="CME", name="Live Cattle", multiplier="", delayed=False
    ),
    "M2K": FutureSymbol(
        symbol="M2K",
        exchange="CME",
        name="Micro E-Mini Russell 2000 Index",
        multiplier="",
        delayed=False,
    ),
    "M6A": FutureSymbol(
        symbol="M6A",
        exchange="CME",
        name="Australian dollar",
        multiplier="",
        delayed=False,
    ),
    "M6B": FutureSymbol(
        symbol="M6B", exchange="CME", name="British pound", multiplier="", delayed=False
    ),
    "M6E": FutureSymbol(
        symbol="M6E",
        exchange="CME",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "MBT": FutureSymbol(
        symbol="MBT", exchange="CME", name="Micro Bitcoin", multiplier="", delayed=False
    ),
    "MCD": FutureSymbol(
        symbol="MCD",
        exchange="CME",
        name="Canadian dollar",
        multiplier="",
        delayed=False,
    ),
    "MES": FutureSymbol(
        symbol="MES",
        exchange="CME",
        name="Micro E-Mini S&P 500 Stock Price Index",
        multiplier="",
        delayed=False,
    ),
    "MET": FutureSymbol(
        symbol="MET", exchange="CME", name="Micro Ether", multiplier="", delayed=False
    ),
    "MIA": FutureSymbol(
        symbol="MIA",
        exchange="CME",
        name="Miami Housing Index",
        multiplier="",
        delayed=False,
    ),
    "MIR": FutureSymbol(
        symbol="MIR", exchange="CME", name="Indian Rupee", multiplier="", delayed=False
    ),
    "MNH": FutureSymbol(
        symbol="MNH",
        exchange="CME",
        name="United States dollar",
        multiplier="",
        delayed=False,
    ),
    "MNQ": FutureSymbol(
        symbol="MNQ",
        exchange="CME",
        name="Micro E-Mini Nasdaq-100 Index",
        multiplier="",
        delayed=False,
    ),
    "MSF": FutureSymbol(
        symbol="MSF", exchange="CME", name="Swiss franc", multiplier="", delayed=False
    ),
    "NKD": FutureSymbol(
        symbol="NKD",
        exchange="CME",
        name="Dollar Denominated Nikkei 225 Index",
        multiplier="",
        delayed=False,
    ),
    "NOK": FutureSymbol(
        symbol="NOK",
        exchange="CME",
        name="Norwegian krone",
        multiplier="",
        delayed=False,
    ),
    "NQ": FutureSymbol(
        symbol="NQ",
        exchange="CME",
        name="E-mini NASDAQ 100",
        multiplier="",
        delayed=False,
    ),
    "NYM": FutureSymbol(
        symbol="NYM",
        exchange="CME",
        name="New York Housing Index",
        multiplier="",
        delayed=False,
    ),
    "PJY": FutureSymbol(
        symbol="PJY", exchange="CME", name="British pound", multiplier="", delayed=False
    ),
    "PLN": FutureSymbol(
        symbol="PLN", exchange="CME", name="Polish zloty", multiplier="", delayed=False
    ),
    "PSF": FutureSymbol(
        symbol="PSF", exchange="CME", name="British pound", multiplier="", delayed=False
    ),
    "RF": FutureSymbol(
        symbol="RF",
        exchange="CME",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "RMB": FutureSymbol(
        symbol="RMB",
        exchange="CME",
        name="CME Chinese Renminbi in US Dollar Cross Rate",
        multiplier="",
        delayed=False,
    ),
    "RME": FutureSymbol(
        symbol="RME",
        exchange="CME",
        name="CME Chinese Renminbi in Euro Cross Rate",
        multiplier="",
        delayed=False,
    ),
    "RP": FutureSymbol(
        symbol="RP",
        exchange="CME",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "RS1": FutureSymbol(
        symbol="RS1",
        exchange="CME",
        name="E-mini Russell 1000 Index Futures",
        multiplier="",
        delayed=False,
    ),
    "RSG": FutureSymbol(
        symbol="RSG",
        exchange="CME",
        name="E-mini Russell 1000 Growth Index Futures",
        multiplier="",
        delayed=False,
    ),
    "RSV": FutureSymbol(
        symbol="RSV",
        exchange="CME",
        name="E-Mini Russell 1000 Value Index Futures",
        multiplier="",
        delayed=False,
    ),
    "RTY": FutureSymbol(
        symbol="RTY",
        exchange="CME",
        name="E-mini Russell 2000 Index",
        multiplier="",
        delayed=False,
    ),
    "RY": FutureSymbol(
        symbol="RY",
        exchange="CME",
        name="European Monetary Union Euro",
        multiplier="",
        delayed=False,
    ),
    "SPXDIVAN": FutureSymbol(
        symbol="SPXDIVAN",
        exchange="CME",
        name="S&P 500 Dividend Points (Annual) Index",
        multiplier="",
        delayed=False,
    ),
    "SDG": FutureSymbol(
        symbol="SDG",
        exchange="CME",
        name="San Diego Housing Index",
        multiplier="",
        delayed=False,
    ),
    "SEK": FutureSymbol(
        symbol="SEK", exchange="CME", name="Swedish krona", multiplier="", delayed=False
    ),
    "SFR": FutureSymbol(
        symbol="SFR",
        exchange="CME",
        name="San Francisco Housing Index",
        multiplier="",
        delayed=False,
    ),
    "SGX": FutureSymbol(
        symbol="SGX",
        exchange="CME",
        name="S&P 500 / Citigroup Growth Index",
        multiplier="",
        delayed=False,
    ),
    "SIR": FutureSymbol(
        symbol="SIR", exchange="CME", name="Indian Rupee", multiplier="", delayed=False
    ),
    "SJY": FutureSymbol(
        symbol="SJY", exchange="CME", name="Swiss franc", multiplier="", delayed=False
    ),
    "SMC": FutureSymbol(
        symbol="SMC",
        exchange="CME",
        name="E-Mini S&P SmallCap 600 Futures",
        multiplier="",
        delayed=False,
    ),
    "SONIA": FutureSymbol(
        symbol="SONIA",
        exchange="CME",
        name="Sterling Overnight Index Average",
        multiplier="",
        delayed=False,
    ),
    "SOFR1": FutureSymbol(
        symbol="SOFR1",
        exchange="CME",
        name="Secured Overnight Financing Rate 1-month average of rates",
        multiplier="",
        delayed=False,
    ),
    "SOFR3": FutureSymbol(
        symbol="SOFR3",
        exchange="CME",
        name="Secured Overnight Financing Rate 3-month average of rates",
        multiplier="",
        delayed=False,
    ),
    "SVX": FutureSymbol(
        symbol="SVX",
        exchange="CME",
        name="S&P 500 / Citigroup Value Index",
        multiplier="",
        delayed=False,
    ),
    "TBF3": FutureSymbol(
        symbol="TBF3",
        exchange="CME",
        name="13-Week US Treasury Bill",
        multiplier="",
        delayed=False,
    ),
    "WDC": FutureSymbol(
        symbol="WDC",
        exchange="CME",
        name="Washington DC Housing Index",
        multiplier="",
        delayed=False,
    ),
    "IXB": FutureSymbol(
        symbol="IXB",
        exchange="CME",
        name="Materials Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXE": FutureSymbol(
        symbol="IXE",
        exchange="CME",
        name="Energy Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXM": FutureSymbol(
        symbol="IXM",
        exchange="CME",
        name="Financial Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXI": FutureSymbol(
        symbol="IXI",
        exchange="CME",
        name="Industrial Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXT": FutureSymbol(
        symbol="IXT",
        exchange="CME",
        name="Technology Select Sector Index -",
        multiplier="",
        delayed=False,
    ),
    "IXR": FutureSymbol(
        symbol="IXR",
        exchange="CME",
        name="Consumer Staples Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXRE": FutureSymbol(
        symbol="IXRE",
        exchange="CME",
        name="Real Estate Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXU": FutureSymbol(
        symbol="IXU",
        exchange="CME",
        name="Utilities Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXV": FutureSymbol(
        symbol="IXV",
        exchange="CME",
        name="Health Care Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "IXY": FutureSymbol(
        symbol="IXY",
        exchange="CME",
        name="Consumer Discretionary Select Sector Index",
        multiplier="",
        delayed=False,
    ),
    "LRC30APR": FutureSymbol(
        symbol="LRC30APR",
        exchange="NYBOT",
        name="ICE U.S. Conforming 30-yr Fixed Mortgage Rate Lock Weighted APR",
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
        symbol="CL",
        exchange="NYMEX",
        name="Light Sweet Crude Oil",
        multiplier="",
        delayed=False,
    ),
    "HH": FutureSymbol(
        symbol="HH",
        exchange="NYMEX",
        name="Natural Gas Last Day Financial Futures Index",
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
    "MCL": FutureSymbol(
        symbol="MCL",
        exchange="NYMEX",
        name="Micro WTI Crude Oil",
        multiplier="",
        delayed=False,
    ),
    "MHO": FutureSymbol(
        symbol="MHO",
        exchange="NYMEX",
        name="Micro NY Harbor ULSD",
        multiplier="",
        delayed=False,
    ),
    "MHNG": FutureSymbol(
        symbol="MHNG",
        exchange="NYMEX",
        name="Micro Henry Hub Natural Gas",
        multiplier="",
        delayed=False,
    ),
    "MRB": FutureSymbol(
        symbol="MRB",
        exchange="NYMEX",
        name="Micro RBOB Gasoline",
        multiplier="",
        delayed=False,
    ),
    "NG": FutureSymbol(
        symbol="NG",
        exchange="NYMEX",
        name="Henry Hub Natural Gas",
        multiplier="",
        delayed=False,
    ),
    "PA": FutureSymbol(
        symbol="PA",
        exchange="NYMEX",
        name="Palladium Index",
        multiplier="",
        delayed=False,
    ),
    "PL": FutureSymbol(
        symbol="PL",
        exchange="NYMEX",
        name="Platinum Index",
        multiplier="",
        delayed=False,
    ),
    "QG": FutureSymbol(
        symbol="QG",
        exchange="NYMEX",
        name="NYMEX MINY Natural Gas Index",
        multiplier="",
        delayed=False,
    ),
    "QM": FutureSymbol(
        symbol="QM",
        exchange="NYMEX",
        name="NYMEX MINY Light Sweet Crude Oil",
        multiplier="",
        delayed=False,
    ),
    "RB": FutureSymbol(
        symbol="RB",
        exchange="NYMEX",
        name="NYMEX RBOB Gasoline Index",
        multiplier="",
        delayed=False,
    ),
    "TT": FutureSymbol(
        symbol="TT",
        exchange="NYMEX",
        name="NYMEX Cotton index",
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
} | {
    # NOTABLE MANUAL EXCEPTIONS TO THE ABOVE:
    # IBKR uses the SAME SYMBOL for bitcoin futures and micro bitcoin futures, with
    # the only difference being the multiplier requirement.
    # We distinguish our usable names via /BTC for full and /MBT for micros.
    "BTC": FutureSymbol(
        symbol="BRR",
        exchange="CME",
        multiplier=5,
        name="CME CF Bitcoin Reference Rate",
    ),
    "MBT": FutureSymbol(
        symbol="BRR",
        exchange="CME",
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
