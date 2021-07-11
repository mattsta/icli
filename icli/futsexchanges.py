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
    ),
    "IBXXIBHY": FutureSymbol(
        symbol="IBXXIBHY",
        exchange="CFE",
        name="iBoxx iShares $ High Yield Corporate Bond Index TR",
    ),
    "IBXXIBIG": FutureSymbol(
        symbol="IBXXIBIG",
        exchange="CFE",
        name="iBoxx iShares $ Investment Grade Corporate Bond Index TR",
    ),
    "ACD": FutureSymbol(symbol="ACD", exchange="GLOBEX", name="Australian dollar"),
    "AUD": FutureSymbol(symbol="AUD", exchange="GLOBEX", name="Australian dollar"),
    "AJY": FutureSymbol(symbol="AJY", exchange="GLOBEX", name="Australian dollar"),
    "GBP": FutureSymbol(symbol="GBP", exchange="GLOBEX", name="British pound"),
    "BRE": FutureSymbol(
        symbol="BRE", exchange="GLOBEX", name="Brazilian Real in US Dollars"
    ),
    "CAD": FutureSymbol(symbol="CAD", exchange="GLOBEX", name="Canadian dollar"),
    "CZK": FutureSymbol(symbol="CZK", exchange="GLOBEX", name="Czech koruna"),
    "EUR": FutureSymbol(
        symbol="EUR", exchange="GLOBEX", name="European Monetary Union Euro"
    ),
    "ECK": FutureSymbol(symbol="ECK", exchange="GLOBEX", name="Czech koruna"),
    "GE": FutureSymbol(symbol="GE", exchange="GLOBEX", name="EURODOLLARS"),
    "EHF": FutureSymbol(symbol="EHF", exchange="GLOBEX", name="Hungarian forint"),
    "EM": FutureSymbol(
        symbol="EM", exchange="GLOBEX", name="1 Month LIBOR (Int. Rate)"
    ),
    "EPZ": FutureSymbol(symbol="EPZ", exchange="GLOBEX", name="Polish zloty"),
    "GF": FutureSymbol(symbol="GF", exchange="GLOBEX", name="Feeder Cattle"),
    "GSCI": FutureSymbol(symbol="GSCI", exchange="GLOBEX", name="S&P-GSCI Index"),
    "HUF": FutureSymbol(symbol="HUF", exchange="GLOBEX", name="Hungarian forint"),
    "JPY": FutureSymbol(symbol="JPY", exchange="GLOBEX", name="Japanese yen"),
    "LE": FutureSymbol(symbol="LE", exchange="GLOBEX", name="Live Cattle"),
    "HE": FutureSymbol(symbol="HE", exchange="GLOBEX", name="Lean Hogs"),
    "MXP": FutureSymbol(symbol="MXP", exchange="GLOBEX", name="Mexican Peso"),
    "NZD": FutureSymbol(symbol="NZD", exchange="GLOBEX", name="New Zealand dollar"),
    "NKD": FutureSymbol(symbol="NKD", exchange="GLOBEX", name="NIKKEI 225"),
    "PLN": FutureSymbol(symbol="PLN", exchange="GLOBEX", name="Polish zloty"),
    "ZAR": FutureSymbol(symbol="ZAR", exchange="GLOBEX", name="South African Rand"),
    "RF": FutureSymbol(
        symbol="RF", exchange="GLOBEX", name="European Monetary Union Euro"
    ),
    "RP": FutureSymbol(
        symbol="RP", exchange="GLOBEX", name="European Monetary Union Euro"
    ),
    "RUR": FutureSymbol(
        symbol="RUR", exchange="GLOBEX", name="Russian Ruble in US Dollars"
    ),
    "RY": FutureSymbol(
        symbol="RY", exchange="GLOBEX", name="European Monetary Union Euro"
    ),
    "CHF": FutureSymbol(symbol="CHF", exchange="GLOBEX", name="Swiss franc"),
    "SPX": FutureSymbol(symbol="SPX", exchange="GLOBEX", name="S&P 500 Stock Index"),
    "ETHUSDRR": FutureSymbol(
        symbol="ETHUSDRR",
        exchange="CMECRYPTO",
        name="CME CF Ether-Dollar Reference Rate",
    ),
    "AIGCI": FutureSymbol(
        symbol="AIGCI", exchange="ECBOT", name="Bloomberg Commodity Index"
    ),
    "B1U": FutureSymbol(
        symbol="B1U",
        exchange="ECBOT",
        name="30-Year Deliverable Interest Rate Swap Futures",
    ),
    "AC": FutureSymbol(symbol="AC", exchange="ECBOT", name="Ethanol -CME"),
    "F1U": FutureSymbol(
        symbol="F1U",
        exchange="ECBOT",
        name="5-Year Deliverable Interest Rate Swap Futures",
    ),
    "KE": FutureSymbol(
        symbol="KE", exchange="ECBOT", name="Hard Red Winter Wheat -KCBOT-"
    ),
    "LIT": FutureSymbol(
        symbol="LIT", exchange="ECBOT", name="2-Year Eris Swap Futures"
    ),
    "LIW": FutureSymbol(
        symbol="LIW", exchange="ECBOT", name="5-Year Eris Swap Futures"
    ),
    "MYM": FutureSymbol(symbol="MYM", exchange="ECBOT", name="Micro E-mini DJIA"),
    "N1U": FutureSymbol(
        symbol="N1U",
        exchange="ECBOT",
        name="10-Year Deliverable Interest Rate Swap Futures",
    ),
    "DJUSRE": FutureSymbol(
        symbol="DJUSRE", exchange="ECBOT", name="Dow Jones US Real Estate Index"
    ),
    "T1U": FutureSymbol(
        symbol="T1U",
        exchange="ECBOT",
        name="2-Year Deliverable Interest Rate Swap Futures",
    ),
    "TN": FutureSymbol(symbol="TN", exchange="ECBOT", name="10-YR T-NOTES"),
    "UB": FutureSymbol(symbol="UB", exchange="ECBOT", name="Ultra T-BONDS"),
    "YC": FutureSymbol(symbol="YC", exchange="ECBOT", name="Mini Sized Corn Futures"),
    "YK": FutureSymbol(
        symbol="YK", exchange="ECBOT", name="Mini Sized Soybean Futures"
    ),
    "YW": FutureSymbol(symbol="YW", exchange="ECBOT", name="Mini Sized Wheat Futures"),
    "YM": FutureSymbol(symbol="YM", exchange="ECBOT", name="MINI DJIA"),
    "Z3N": FutureSymbol(symbol="Z3N", exchange="ECBOT", name="3-YR TREAS."),
    "ZB": FutureSymbol(symbol="ZB", exchange="ECBOT", name="30-year T-BONDS"),
    "ZC": FutureSymbol(symbol="ZC", exchange="ECBOT", name="Corn Futures"),
    "ZF": FutureSymbol(symbol="ZF", exchange="ECBOT", name="5-YR TREAS."),
    "ZL": FutureSymbol(symbol="ZL", exchange="ECBOT", name="Soybean Oil Futures"),
    "ZM": FutureSymbol(symbol="ZM", exchange="ECBOT", name="Soybean Meal Futures"),
    "ZN": FutureSymbol(symbol="ZN", exchange="ECBOT", name="10-YR TREAS."),
    "ZO": FutureSymbol(symbol="ZO", exchange="ECBOT", name="Oat Futures"),
    "ZQ": FutureSymbol(symbol="ZQ", exchange="ECBOT", name="30 Day Federal Funds"),
    "ZR": FutureSymbol(symbol="ZR", exchange="ECBOT", name="Rough Rice Futures"),
    "ZS": FutureSymbol(symbol="ZS", exchange="ECBOT", name="Soybean Futures"),
    "ZT": FutureSymbol(symbol="ZT", exchange="ECBOT", name="2-YR TREAS."),
    "ZW": FutureSymbol(symbol="ZW", exchange="ECBOT", name="Wheat Futures"),
    "BQX": FutureSymbol(
        symbol="BQX", exchange="GLOBEX", name="CME E-Mini NASDAQ Biotechnology"
    ),
    "BOS": FutureSymbol(symbol="BOS", exchange="GLOBEX", name="Boston Housing Index"),
    "CB": FutureSymbol(
        symbol="CB", exchange="GLOBEX", name="CME Cash-Settled Butter Futures"
    ),
    "CHI": FutureSymbol(symbol="CHI", exchange="GLOBEX", name="Chicago Housing Index"),
    "CLP": FutureSymbol(symbol="CLP", exchange="GLOBEX", name="Chilean peso"),
    "CJY": FutureSymbol(symbol="CJY", exchange="GLOBEX", name="Canadian dollar"),
    "CNH": FutureSymbol(symbol="CNH", exchange="GLOBEX", name="United States dollar"),
    "CSC": FutureSymbol(symbol="CSC", exchange="GLOBEX", name="Cheese"),
    "CUS": FutureSymbol(
        symbol="CUS", exchange="GLOBEX", name="Housing Index Composite"
    ),
    "DA": FutureSymbol(symbol="DA", exchange="GLOBEX", name="MILK CLASS III INDEX"),
    "DEN": FutureSymbol(symbol="DEN", exchange="GLOBEX", name="Denver Housing Index"),
    "DY": FutureSymbol(symbol="DY", exchange="GLOBEX", name="CME DRY WHEY INDEX"),
    "E7": FutureSymbol(
        symbol="E7", exchange="GLOBEX", name="European Monetary Union Euro"
    ),
    "EAD": FutureSymbol(
        symbol="EAD", exchange="GLOBEX", name="European Monetary Union Euro"
    ),
    "ECD": FutureSymbol(
        symbol="ECD", exchange="GLOBEX", name="European Monetary Union Euro"
    ),
    "EMD": FutureSymbol(
        symbol="EMD", exchange="GLOBEX", name="E-mini S&P Midcap 400 Futures"
    ),
    "NIY": FutureSymbol(
        symbol="NIY", exchange="GLOBEX", name="Yen Denominated Nikkei 225 Index"
    ),
    "ES": FutureSymbol(symbol="ES", exchange="GLOBEX", name="MINI-S&P 500"),
    "SPXESUP": FutureSymbol(
        symbol="SPXESUP", exchange="GLOBEX", name="E-mini S&P 500 ESG"
    ),
    "GDK": FutureSymbol(
        symbol="GDK", exchange="GLOBEX", name="Class IV Milk - 200k lbs"
    ),
    "NF": FutureSymbol(symbol="NF", exchange="GLOBEX", name="NON FAT DRY MILK INDEX"),
    "IBAA": FutureSymbol(symbol="IBAA", exchange="GLOBEX", name="Bovespa Index - USD"),
    "ILS": FutureSymbol(
        symbol="ILS", exchange="GLOBEX", name="Israeli Shekel in US Dollar"
    ),
    "J7": FutureSymbol(symbol="J7", exchange="GLOBEX", name="Japanese yen"),
    "KRW": FutureSymbol(symbol="KRW", exchange="GLOBEX", name="Korean Won"),
    "LAV": FutureSymbol(
        symbol="LAV", exchange="GLOBEX", name="Las Vegas Housing Index"
    ),
    "LAX": FutureSymbol(
        symbol="LAX", exchange="GLOBEX", name="Los Angeles Housing Index"
    ),
    "LB": FutureSymbol(symbol="LB", exchange="GLOBEX", name="Random Length Lumber"),
    "M2K": FutureSymbol(
        symbol="M2K", exchange="GLOBEX", name="Micro E-mini Russell 2000"
    ),
    "M6A": FutureSymbol(symbol="M6A", exchange="GLOBEX", name="Australian dollar"),
    "M6B": FutureSymbol(symbol="M6B", exchange="GLOBEX", name="British pound"),
    "M6E": FutureSymbol(
        symbol="M6E", exchange="GLOBEX", name="European Monetary Union Euro"
    ),
    "MCD": FutureSymbol(symbol="MCD", exchange="GLOBEX", name="Canadian dollar"),
    "MES": FutureSymbol(symbol="MES", exchange="GLOBEX", name="Micro E-mini S&P 500"),
    "MIA": FutureSymbol(symbol="MIA", exchange="GLOBEX", name="Miami Housing Index"),
    "MIR": FutureSymbol(symbol="MIR", exchange="GLOBEX", name="Indian Rupee"),
    "MJY": FutureSymbol(symbol="MJY", exchange="GLOBEX", name="Japanese yen"),
    "MNH": FutureSymbol(symbol="MNH", exchange="GLOBEX", name="United States dollar"),
    "MNQ": FutureSymbol(
        symbol="MNQ", exchange="GLOBEX", name="Micro E-mini NASDAQ-100"
    ),
    "MSF": FutureSymbol(symbol="MSF", exchange="GLOBEX", name="Swiss franc"),
    "NOK": FutureSymbol(symbol="NOK", exchange="GLOBEX", name="Norwegian krone"),
    "NQ": FutureSymbol(symbol="NQ", exchange="GLOBEX", name="NASDAQ E-MINI"),
    "NYM": FutureSymbol(symbol="NYM", exchange="GLOBEX", name="New York Housing Index"),
    "PJY": FutureSymbol(symbol="PJY", exchange="GLOBEX", name="British pound"),
    "PSF": FutureSymbol(symbol="PSF", exchange="GLOBEX", name="British pound"),
    "RMB": FutureSymbol(
        symbol="RMB",
        exchange="GLOBEX",
        name="CME Chinese Renminbi in US Dollar Cross Rate",
    ),
    "RME": FutureSymbol(
        symbol="RME", exchange="GLOBEX", name="CME Chinese Renminbi in Euro Cross Rate"
    ),
    "RS1": FutureSymbol(
        symbol="RS1", exchange="GLOBEX", name="E-mini Russell 1000 Index Futures"
    ),
    "RSG": FutureSymbol(
        symbol="RSG", exchange="GLOBEX", name="E-mini Russell 1000 Growth Index Futures"
    ),
    "RSV": FutureSymbol(
        symbol="RSV", exchange="GLOBEX", name="E-Mini Russell 1000 Value Index Futures"
    ),
    "RTY": FutureSymbol(
        symbol="RTY", exchange="GLOBEX", name="E-Mini Russell 2000 Index"
    ),
    "SPXDIVAN": FutureSymbol(
        symbol="SPXDIVAN",
        exchange="GLOBEX",
        name="S&P 500 Dividend Points (Annual) Index",
    ),
    "SDG": FutureSymbol(
        symbol="SDG", exchange="GLOBEX", name="San Diego Housing Index"
    ),
    "SEK": FutureSymbol(symbol="SEK", exchange="GLOBEX", name="Swedish krona"),
    "SFR": FutureSymbol(
        symbol="SFR", exchange="GLOBEX", name="San Francisco Housing Index"
    ),
    "SGX": FutureSymbol(
        symbol="SGX", exchange="GLOBEX", name="S&P 500 / Citigroup Growth Index"
    ),
    "SIR": FutureSymbol(symbol="SIR", exchange="GLOBEX", name="Indian Rupee"),
    "SJY": FutureSymbol(symbol="SJY", exchange="GLOBEX", name="Swiss franc"),
    "SMC": FutureSymbol(
        symbol="SMC", exchange="GLOBEX", name="E-Mini S&P SmallCap 600 Futures"
    ),
    "SONIA": FutureSymbol(
        symbol="SONIA", exchange="GLOBEX", name="Sterling Overnight Index Average"
    ),
    "SOFR1": FutureSymbol(
        symbol="SOFR1",
        exchange="GLOBEX",
        name="Secured Overnight Financing Rate 1-month average of rates",
    ),
    "SOFR3": FutureSymbol(
        symbol="SOFR3",
        exchange="GLOBEX",
        name="Secured Overnight Financing Rate 3-month average of rates",
    ),
    "SVX": FutureSymbol(
        symbol="SVX", exchange="GLOBEX", name="S&P 500 / Citigroup Value Index"
    ),
    "WDC": FutureSymbol(
        symbol="WDC", exchange="GLOBEX", name="Washington DC Housing Index"
    ),
    "IXB": FutureSymbol(
        symbol="IXB", exchange="GLOBEX", name="Materials Select Sector Index"
    ),
    "IXE": FutureSymbol(
        symbol="IXE", exchange="GLOBEX", name="Energy Select Sector Index"
    ),
    "IXM": FutureSymbol(
        symbol="IXM", exchange="GLOBEX", name="Financial Select Sector Index"
    ),
    "IXI": FutureSymbol(
        symbol="IXI", exchange="GLOBEX", name="Industrial Select Sector Index"
    ),
    "IXT": FutureSymbol(
        symbol="IXT", exchange="GLOBEX", name="Technology Select Sector Index -"
    ),
    "IXR": FutureSymbol(
        symbol="IXR", exchange="GLOBEX", name="Consumer Staples Select Sector Index"
    ),
    "IXRE": FutureSymbol(
        symbol="IXRE", exchange="GLOBEX", name="Real Estate Select Sector Index"
    ),
    "IXU": FutureSymbol(
        symbol="IXU", exchange="GLOBEX", name="Utilities Select Sector Index"
    ),
    "IXV": FutureSymbol(
        symbol="IXV", exchange="GLOBEX", name="Health Care Select Sector Index"
    ),
    "IXY": FutureSymbol(
        symbol="IXY",
        exchange="GLOBEX",
        name="Consumer Discretionary Select Sector Index",
    ),
    "CC": FutureSymbol(symbol="CC", exchange="NYBOT", name="Cocoa NYBOT"),
    "CT": FutureSymbol(symbol="CT", exchange="NYBOT", name="Cotton No. 2"),
    "DX": FutureSymbol(symbol="DX", exchange="NYBOT", name="NYBOT US Dollar FX"),
    "NYFANG": FutureSymbol(symbol="NYFANG", exchange="NYBOT", name="NYSE FANG+ Index"),
    "KC": FutureSymbol(symbol="KC", exchange="NYBOT", name='Coffee "C"'),
    "OJ": FutureSymbol(symbol="OJ", exchange="NYBOT", name='FC Orange Juice "A"'),
    "RS": FutureSymbol(symbol="RS", exchange="NYBOT", name="Canola"),
    "SB": FutureSymbol(symbol="SB", exchange="NYBOT", name="Sugar No. 11"),
    "SF": FutureSymbol(symbol="SF", exchange="NYBOT", name="Sugar #16 112000 lbs"),
    "ALI": FutureSymbol(symbol="ALI", exchange="NYMEX", name="NYMEX Aluminum Index"),
    "BB": FutureSymbol(
        symbol="BB", exchange="NYMEX", name="NYMEX Brent Financial Futures Index"
    ),
    "BZ": FutureSymbol(
        symbol="BZ", exchange="NYMEX", name="Brent Crude Oil - Last Day"
    ),
    "CL": FutureSymbol(symbol="CL", exchange="NYMEX", name="Crude oil"),
    "GC": FutureSymbol(symbol="GC", exchange="NYMEX", name="Gold"),
    "HG": FutureSymbol(symbol="HG", exchange="NYMEX", name="Copper"),
    "HH": FutureSymbol(
        symbol="HH", exchange="NYMEX", name="Nautral Gas Last Day Financial  Future"
    ),
    "HO": FutureSymbol(symbol="HO", exchange="NYMEX", name="Heating Oil"),
    "HP": FutureSymbol(
        symbol="HP",
        exchange="NYMEX",
        name="Natural Gas Penultimate Financial Futures Index",
    ),
    "HRC": FutureSymbol(
        symbol="HRC", exchange="NYMEX", name="Hot-Rolled Coil Steel Index"
    ),
    "MGC": FutureSymbol(symbol="MGC", exchange="NYMEX", name="E-Micro Gold"),
    "NG": FutureSymbol(symbol="NG", exchange="NYMEX", name="Natural gas"),
    "PA": FutureSymbol(symbol="PA", exchange="NYMEX", name="NYMEX Palladium Index"),
    "PL": FutureSymbol(symbol="PL", exchange="NYMEX", name="NYMEX Platinum Index"),
    "QC": FutureSymbol(symbol="QC", exchange="NYMEX", name="Copper"),
    "QG": FutureSymbol(symbol="QG", exchange="NYMEX", name="Natural gas E-Mini"),
    "QH": FutureSymbol(symbol="QH", exchange="NYMEX", name="Heating Oil E-Mini"),
    "QI": FutureSymbol(symbol="QI", exchange="NYMEX", name="Silver Mini"),
    "QM": FutureSymbol(symbol="QM", exchange="NYMEX", name="Crude oil E-Mini"),
    "QO": FutureSymbol(symbol="QO", exchange="NYMEX", name="Gold"),
    "QU": FutureSymbol(symbol="QU", exchange="NYMEX", name="Unleaded Gasoline E-Mini"),
    "RB": FutureSymbol(symbol="RB", exchange="NYMEX", name="RBOB Gasoline"),
    "SGC": FutureSymbol(
        symbol="SGC",
        exchange="NYMEX",
        name="Shanghai Gold Exchange Gold Benchmark PM Price Index - CNH Futures",
    ),
    "SGUF": FutureSymbol(
        symbol="SGUF",
        exchange="NYMEX",
        name="Shanghai Gold Exchange Gold Benchmark PM Price Index - USD Futures",
    ),
    "SI": FutureSymbol(symbol="SI", exchange="NYMEX", name="Silver"),
    "TT": FutureSymbol(symbol="TT", exchange="NYMEX", name="NYMEX Cotton index"),
    "UX": FutureSymbol(symbol="UX", exchange="NYMEX", name="NYMEX Uranium Index"),
    "SP": FutureSymbol(symbol="SP", exchange="GLOBEX", name="S&P 500"),
    "SIL": FutureSymbol(symbol="SIL", exchange="NYMEX", name="Silver"),
    "TF": FutureSymbol(symbol="TF", exchange="NYBOT", name="RUSSELL 2000"),
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
