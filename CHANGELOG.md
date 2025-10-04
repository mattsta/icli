# 2025-10-XX - 2.NEXT

## 🚀 Highlights

- **Command-per-file Plugin System**: Refactored commands out of the single 8,000 line `lang.py` file into individual `icli/cmds/[CATEGORY]/[cmd].py` files.
- **Style / Structure Cleanup**: `poetry run ruff check --fix` with Python 3.12 feature levels.
- **Prettier Documetnation**: just ran `prettier -w` on all markdown files.
- **Passes mypy Clean**: `poetry run mypy icli/{cli,cmds/*/*,helpers,orders,instrumentdb,__main__,calc}.py`

---

# 2025-09-28 - 2.0.0

This update introduces new automation capabilities, significant stability improvements, fixes, and improvements for correctness and performance written over thousands of hours from the past ten months all in one giant update.

## 🚀 Highlights

- **Automated Trading with Predicate Engine**: A powerful new `ifthen` engine for creating complex, data-driven trading rules in real-time.
- **Guaranteed Price Compliance**: A new `instrumentdb` ensures all orders use correct price increments, eliminating rejections from incorrect price ticks.
- **Massive `buy` Command Overhaul**: The `buy` command is now a full-fledged order parser with support for brackets, trailing stops, and scale/ladder orders.
- **Enhanced Stability and Performance**: Major refactoring for better caching, error handling, and responsiveness.
- **mypy clean**: Guaranteed mypy clean using: poetry run mypy icli/{cli,lang,helpers,orders,instrumentdb,**main**,calc}.py

## ✨ Major New Features

- **Instrument Database (`instrumentdb`)**:
  - Automatically fetches and caches market rules for price increments, decimals, and valid exchanges.
  - Ensures all orders are compliant with exchange rules, preventing rejections due to incorrect price ticks.
    - e.g. equities trade in $0.01 increments, but options can be $0.01 or $0.05 or $0.10 or $0.25 increments, while some futures are $0.001 or $0.002 or $0.25 or $5 or 0.00390625 or 0.03125 (and more!) increments, etc.
    - Have you ever had an order rejected because the price was "wrong?" Well, never again!
  - Provides a centralized `comply` API for automatically rounding prices up, down, or to the nearest valid price increment for any trading instrument.
  - Though, it's your fault if your instruments are trading a prices like `110.625000 ±0.015625` so the interface may break columns in weird ways.

- **Predicate Engine (`ifthen`)**:
  - Create complex, conditional trading logic using a simple `if...then...` syntax.
    - `if AAPL bid <= AAPL low: buy AAPL 100 LMT` — if AAPL hits the LOD, buy 100 shares at the current price.
    - `if AAPL { bid <= ask }: say AAPL price at AAPL.bid` — use factored symbols for cleaner syntax.
    - `if AAPL ask >= AAPL high - 0.75: say AAPL near HOD again` — supports arbitrary math against live values
    - `if :23 delta >= 0.50: say delta going up` — use positional quote aliases.
    - `if AAPL last >= (AAPL high + AAPL low) / 2: say AAPL is over half way between high and low` — supports arbitrary math.
    - `if :23 { bid >= low + 5% }: say row 23 is 5% higher than the low again` — supports percentage-based math.
    - `if :23 { ema:delta 300 > ema:delta 900 }: say 5 minute delta rising over 15 minute delta` - trigger events on EMA of deltas (or ema:price, ema:iv too)
    - symbols can also be "factored out" like `if AAPL { bid <= ask }: say AAPL price at AAPL.bid` - avoids repeating `if AAPL bid <= AAPL ask: say done` etc
  - Conditions are evaluated in real-time _on every ticker update_.
  - An `ifthen` statement remains live until its condition matches, then the result executes and the statement is removed from live updates.
  - You can create an algo system by running `ifthen` statements in a loop using the `ifgroup :load <filename>` command where filename is an `ifthen` DSL for defining an initial start predicate, then when it matches, _another _ predicate is scheduled, and these can be placed in arbitrary tree (OTO) or peer (OCO) groups.
  - the fields available are: high, low, open, close, last, mid/midpoint/live, atr, vwap, iv, delta, theta, gamma
  - Can consume external data from WebSocket feeds via the new **Algo Binder**.
  - Manage predicates with `iflist`, `ifrm`, `ifclear`, and `ifgroup` commands.

- **Order Manager (`ordermgr`)**:
  - New local system for tracking executed positions and trades.
  - Provides a foundation for advanced portfolio analysis and reporting (`report` command).

- **Background Task & Scheduling System (`bgtask`)**:
  - Replaced ad-hoc scheduler with a robust `BGTask` system.
  - Manage scheduled commands with `sched-add`, `sched-list`, `sched-cancel`.
  - Manage background tasks with `tasklist`, `taskcancel`.

- **Algo Binder**:
  - New module to consume external data feeds via WebSockets.
  - Integrates external algorithmic signals directly into the `ifthen` predicate engine.

- **Paper Trading Log**:
  - A new `paper` command for session-based paper trading simulation.

## 🛠️ Command Enhancements

- **`buy` Command**:
  - Rewritten to use a full order language parser (`orderlang`), allowing for flexible syntax.
    - `buy` command input is now a full parser ([`orderlang`](https://github.com/mattsta/tradeapis/blob/main/tradeapis/orderlang.py) in tradeapis) instead of just 3 command fields — we have much more flexibility for parameter order, types of information (attached take-profit, stop-loss either one or both and optional algo overrides for each attached order), and config options now (trailing stop orders with initial stop, trail, limit prices all in the `buy` command).
  - Now supports complex orders with attached brackets (take-profit/stop-loss), trailing stops, and scale/ladder orders.
  - Added support for more IBKR order types, including `TRAIL LIMIT` and `SNAP` orders.
  - Price-following logic is now event-driven for faster, more responsive fills.
  - Enhanced `buy ... preview` output with detailed margin impact, commission estimates, and risk/reward analysis.

- **`straddle` Command**:
  - Now a powerful spread creation tool, not just for orders but for adding quotes.
  - Supports straddles, strangles, vertical spreads, and iron-condor-like structures with a flexible syntax.
  - Example: `straddle /ES v c 10 20 v p -10 -20` creates a complex spread quote.
  - Automatically flips legs when selling a long-defined spread.
  - `straddle /ES 20` would add a quote for a straddle (long call, long put) ±20 points from live /ES ATM
  - `straddle /ES 0` would give you an ATM strangle
  - `straddle /ES vertical put -10 -20` would give you a vertical put with the long leg -10 points from ATM and the short leg -20 points from ATM (e.g. /ES 5000 with vertical put -10 -20 would add a quote for BUY PUT 4980, SELL PUT 4960)
  - `straddle /ES vertical call 10 20` creates a spread quote for buy call +10 ATM and sell call +20 ATM.
  - You can also combine the syntax in any combination (also with shorthand): `straddle /ES v c 10 20 v p -10 -20` would create a quote for (BUY CALL +10 ATM, SELL CALL +20 ATM, BUY PUT -10 ATM, SELL PUT -20 ATM) all as a single spread.

- **`clear` Command**:
  - New subcommands for cleaning up the quote list: `today`, `expired`, `unused`, `options`, `noquote`.
  - `clear today` for removing all quotes expiring today or older
  - `clear expired` for removing all quotes older than today
  - `clear unused` for removing all single leg option quotes not used by spread quotes
  - `clear options` for removing all options quotes

- **`align` Command**:
  - New command to generate a family of related spread quotes at once (e.g., ATM strangle, straddle, and vertical spreads).
  - `align /ES` by default gives you: ATM strangle, 10 point wide straddle, call spread +10 ATM 20 points wide, put spread -10 ATM 20 points wide.
  - The ATM point offset and width offset can be adjusted as parameters.

- **`set` / `unset` Commands**:
  - `set dte` now accepts day names (e.g., `thursday`) to automatically calculate days to expiration.
  - contract discovery commands, by default, use the nearest contract expiration, but you can alter how far the automatic quotes expire with a global setting of `set dte N` where `N` is days away from today. `set dte` can also accept a day name as a word (`set dte thursday`) and it will figure out how many days away to use for `N`

- **`info` Command**:
  - Output is now significantly more detailed, including live stats, EMAs, ATRs, greeks analysis, and quote flow metrics.

- **`positions` Command**:
  - Improved display with better formatting and PnL per share.
  - Now identifies and groups potential spreads.

- **`orders` Command**:
  - Improved display with more details and better sorting.

- **`executions` Command**:
  - Now groups executions by trading day for clearer daily P&L analysis.

- **`daydumper` Command**:
  - New command to generate real time price alerts when crossing historical points of interest and saves historical bar data for a symbol to disk.

- **`colorset` / `colorsload`**:
  - New commands to manage and persist UI color themes.

- **`report` Command**:
  - New command for advanced portfolio analysis and reporting, powered by `ordermgr`.

## ⚙️ Core Improvements & Stability

- **Price & Math Precision**:
  - All internal price calculations now use Python's `Decimal` type for arbitrary precision, eliminating floating-point errors.

- **Data Caching & Qualification**:
  - The `qualify` method for contracts is now much more robust, with better caching, corruption checks, and batching to avoid API rate limits.
  - Contract cache expiration is now smarter (e.g., 5 days for futures, 90 for others).
  - Option chains can be fetched from Tradier (if API key is provided) for a massive speedup over IBKR's API.

- **Quote & Toolbar Display**:
  - Quote logic is now encapsulated in a new `ITicker` class, which augments per-instrument price history with extra live data: EMAs, ATRs, greeks, and more.
  - Quote sorting in the toolbar is improved to handle more security types and sort bags more intelligently.
  - The toolbar now shows more PnL percentage breakdowns (`RealizedPnL%`, `UnrealizedPnL%`, `TotalPnL%`).
  - The toolbar refresh count now distinguishes between total updates and updates since the last reconnect.
  - Toolbar now displays correct digit resolution for all instruments using `instrumentdb`

- **Error Handling**:
  - A new `DuplicateMessageHandler` suppresses repeated API error messages for a cleaner console.
  - Text-to-speech error logging is now throttled to prevent log spam.

- **Code Quality & Dependencies**:
  - Major refactoring into new modules like `helpers.py`, `utils.py`, and `instrumentdb.py`.
  - The codebase now passes `mypy` type checks, improving reliability.
  - Replaced the unmaintained `pendulum` library with `whenever` and `pytz` for robust timezone handling.
  - Now uses `python-dotenv` to load `.env.icli` into `os.environ` at startup.

- **Logging**:
  - Log files are now organized by date (`runlogs/YEAR/MONTH/`).
  - Log filenames include client ID and a full timestamp for easier debugging.

## 🐛 Bug Fixes

- **Calculator**: Fixed `round` function to correctly handle its arguments. Fixed `positionlookup` to handle `None` for bid/ask prices. Added `ac` (average cost) and `abs` (absolute value) functions.
- **Order Modification**: Fixed a critical bug where modifying an order could be rejected by IBKR due to auto-populated fields (e.g., `orderType="IBALGO"`). The new `safeModify` helper prevents this.
- **Futures Options Sorting**: Corrected sorting for futures options to use the full contract month date.
- **Contract Cache**: Added checks to prevent and detect corrupted contract data in the cache.
