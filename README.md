icli: IBKR live trade cli
=========================

`icli` is a command line interface for live trading (or sandbox/paper trading) using IBKR accounts.

The intended use case is for scalping or swing trading, so we prioritize easy of understanding your positions and active orders while removing unnecessary steps when placing orders.

## Demo Replay

Watch the console replay demo below to see a paper trading account where we add some live quotes for stocks and options, remove quotes, place orders to buy and sell, check execution details to view aggregate commission charges, check outstanding order details, and add some live futures quotes too.

[![asciicast](https://asciinema.org/a/424814.svg)](https://asciinema.org/a/424814)

## Overview

Welcome to `icli`! You can use `icli` to manage manual and automated trading using your IBKR account for stocks, futures, currencies, and options.

You can enable audio announcements of trade events if you also run [awwdio](https://github.com/mattsta/awwdio) and provide your `awwdio` address as an environment variable like `ICLI_AWWDIO_URL=http://127.0.0.1:8000 poetry run icli` (macOS only currently and you need to manually install the system speech voice packs in system settings).

There's always a hundred more features we could add, but focus goes mainly towards usability and efficient order entry currently. There's an unreleased in-progress algo trading agent plugin to automatically buy and sell stocks, futures, and/or options based on external signals too. Cleaning those up for public release isn't a priority at the moment unless sponsors would like to step up and motivate us for a wider code release to combine all the feautres into a finished product we can publish.

You can run multiple clients in different terminal windows using unique client ids for each session like `ICLI_CLIENT_ID=4 poetry run icli`. Note: IBKR restricts orders per-client-id, so for example if you place an order under client id 4, the order will not show up under other clients.


Some helpful advanced commands only available in `icli`:

- We have an efficient quote adding system where one command of `add SPY240412{P,C}005{1,2,3}0000` will add live quotes for each of: `SPY240412C00510000`, `SPY240412C00520000`, `SPY240412C00530000`, `SPY240412P00510000`, `SPY240412P00520000`, `SPY240412P00530000`.
  - Then if you want to easily remove those quotes, you can run `rm SPY240412{P,C}005{1,2,3}0000` too. The syntax also supports range population like `add SPXW24041{5..7}{P05135,C05150}000`. Each quote also has a row id, and you can remove quotes by row id individually or using replacement syntax: `rm :31`, `rm :{31..37}`, `rm :{25,27,29}` etc.
- You can run multiple purchases concurrently using the `expand` wrapper like: `expand buy {META,MSFT,NVDA,AMD,AAPL} $15_000 MID` — that command will buy approximately $15,000 worth of _each_ symbol using current midpoint quote prices.
- You can easily empty your portfolio with `evict * -1 0 MID` or immediately sell symbols with current market price caps using `evict MSFT -1 0 AF` etc as well.
  - See `expand?` and `evict?` and `buy?` for more details of how it all works.
- The price format also supports negative prices for shorts or sells, so you can do `buy AAPL 100 MID` to buy then `buy AAPL -100 MID` to sell. You can buy and sell by any combination of price and quantity: `buy MSFT $10_000 AF`, `buy MSFT -10 MKT` etc.
- The price format also doubles as a share count format if you don't include `$`, so `buy AAPL 100 AF` will buy 100 shares of AAPL at the current market price using IBKR's Adaptive Fast order algo type. Price or Quantity values can be positive or negative: `buy AAPL 100 AF`, `buy AAPL $10_000 AF`, `buy AAPL -20 AF`, `buy AAPL -$40_000 AF`. You can append `preview` to `buy` orders to get an estimate of margin impact and other account-based order details too.
- The cli also supports running any commands concurrently or sequentially. This will run both `buy` commands concurrently then show your portfolio holdings after they complete:
  - `buy AAPL 100 AF preview&; buy MSFT 100 AF preview&; ls`.
  - Basically: append `&` to your commands to run them concurrently then split commands with `;`, and you can also run any commands sequentially with just `;` as well like: `ls; ord; bal SMA`
  - Note: these two are equivalent, but the `expand` version is easier if your purchase quantity/amount and algo are the same: `buy AAPL 100 AF preview&; buy MSFT 100 AF preview&` versus `expand buy {MSFT,AAPL} 100 AF prevew`
- We also have a built-in account and market calculator operating in prefix mode:
  - `(/ :BP3 AAPL)` shows how many shares you can buy of AAPL (on 33% margin).
  - `(grow :AF 300)` calculates your AvailableFunds growing by 300%.
  - Caluclation functions can be combined arbitrarily like: `(grow (* AAPL (/ :BP3 AAPL)) 7)`
  - You can also do math on arbitrary symbols if you want to for some reason: `(/ AAPL TSLA)`
  - prices used for symbol math are the live bid/ask midpoint price for each symbol.
  - You can also use row-position details for a live quote value too: `(/ AAPL :18)`.

See below for further account setup and environment variable conditions under the Download section.

For tracking feature updates, check the full commit history and you can always run `?` for listing commands and getting help when running commands. The README is slightly undermaintained and some features are lacking complete documentation outside of the runtime itself, so feel free to suggest updates where useful.


## Features

- allows full trading and data access to your IBKR account using only a CLI
    - note: IBKR doesn't allow all operations from their API, so some operations like money transfers, funding, requesting market data, requesting trading permissions, etc, still need to use the mobile/web apps.
- allows trading as fast as you can type (no need to navigate multiple screens / checks / pre-flight confirmations)
    - in fact, no confirmation for anything. you type it, it happens.
    - forward implication: also allows losing money as fast as you can type
- commands can be entered as any unambiguous prefix
    - e.g. `position` command can be entered as just `p` because it doesn't conflict with any other command
    - but for `qquote` and `quote` commands, so those must be clarified by using `qq` or `qu` at a minimum
- interface shows balances, components of account value, P&L, and other account stats updating in real time
- an excessive amount of attention is given to formatting data for maximum readability and quick understanding at a glance
    - due to density of visible data, it's recommended to run a smaller terminal font size to enable 170+ character wide viewing
- uses console ansi color gradients to show the "goodness" or "badness" of current price quotes
    - bad is red
    - more bad is more red
    - good is green
    - more good is more green
    - really really good is blue (>= 0.98% change)
- helpful CLI prompts for multi-stage operations (thanks to [questionary](https://github.com/tmbo/questionary))
- selects correct class of IBKR contract based on names entered (see: `helpers.py:contractForName()`)
    - futures are prefixed with slashes, as is a norm: `/ES`, `/MES`, `/NQ`, `/MNQ`, etc
    - options are the full OCC symbol (no spaces): `AAPL210716C00155000`
    - future options start with a slash: `/ES210716C05000000`
    - warrants, bonds, bills, forex, etc, aren't currently addressable in the CLI because we didn't decide on a naming convention yet
    - spreads can be entered as a full buy/sell ratio description like:
        - `"bto 1 AAPL210716C00155000 sto 2 AAPL210716C00160000 bto 1 AAPL210716C00165000"`
        - works for adding live quotes (`add "bto ... sto ..."`) and for requesting spread trades using the `spread` command
            - for ordering spreads, you specify the baseline spread per-leg ratio (e.g. butterflies are 1:2:1 as above), then the total spread order is the quantity requested multiplied by each leg ratio (e.g. butterfly 1:2:1 with quantity 100 will order 100 + 200 + 100 = 400 total contracts via a [COB](https://flextrade.com/simplifying-complexity-trading-complex-order-books-in-options-part-1/))
- the positions CLI view also shows matched closing orders for each currently owned symbol
- helper shorthands, like an `EVICT [symbol] [quantity]` command to immediately yeet an entire equity position into a [MidPrice](https://www.interactivebrokers.com/en/index.php?f=36735) order with no extra steps
    - for closing any position, you can enter `-1` as the quantity to use full quantity held
- support for showing depth-of-market requests for a quick glance at market composition (`dom` command)
    - though, IBKR DOM is awful because even if you do pay the extra $300+/month to get full access, it only returns 5 levels on each side. If you want DOM, I'd recommend WeBull's TotalView discount package for $1.99/month and for options use futu/moomoo "Option Order Book Depth" quote pack (which includes real time options quotes) for $3.99/month (though, futu/momo is still pseudo-dom because it just shows top-of-book for each exchange instead of actual depth at any exchange).
- easy order modifications/updates
    - though, IBKR recommends you only modify order quantity and price, otherwise the order should be canceled and placed again if other fields need to be adjusted.
- real time notification of order executions
    - also plays [the best song ever](https://www.youtube.com/watch?v=VwEnx_NMZ4I&t=5s) when any trade is successfully executed as buy or sell
- quick order cancellation—need to bail on one or more orders ASAP before they try to execute? we got u fam
    - `cancel` command with no arguments will bring up a menu for you to select one or more live orders to immediately cancel.
    - `cancel order1 order2 ... orderN` command with arguments will also cancel each requested order id immediately.
        - order ids are viewable in the `orders` command output table
- quick helper to add all waiting orders to your live real time quote view (`oadd`: add all order symbols to quote view)
- you can add individual symbols to the live quote view (equity, option, future, even spreads) (`add [symbols...]`: add symbols to quote view, quoted symbols can be spreads like `add AAPL QQQ "buy 1 AAPL210716C00160000 sell 1 AAPL210716C00155000"` to quote a credit spread and two equity symbols)
- you can also check quotes without adding them to the persistent live updating quote view with `qquote [symbols...]`
    - the individual quote request process is annoyingly slow because of how IBKR reports quotes, but it still works
- view your trade execution history for the current day (`executions` command)
    - includes per-execution commission cost (or credit!)
    - also shows the sum total of your commissions for the day

### Streaming Quote View

#### Stocks / Futures

```haskell
ES    :   4,360.25  ( 1.55%  67.00) (  1.09%   47.25)   4,362.00     4,293.25   4,360.00 x    158   4,360.25 x    112    4,310.25   4,313.00
```

Live stock / etf / etn / futures / etc quotes show:

- quote symbol (`ES`)
- current estimated price (`4,360.25`)
- low: percentage and point amount above low for the session (`( 1.55%  67.00)`)
    - always positive unless symbol is trading at its low price of the day
- close: percentage and point amount from the previous session close (`(  1.09%   47.25)`)
    - the typical "daily price change" value
- session high (`4,362.00`)
- session low (`4,293.25`)
- NBBO bid price x size (`4,360.00 x    158`)
- NBBO ask price x size (`4,360.25 x    112`)
- session open price (`4,310.25`)
- previous session close price (`4,313.00`)

For more detailed level 2 quotes, we'd recommend [WeBull's TotalView integration](https://www.webull.com/activity/get-free-nasdaq-totalview) because they have a special deal with NASDAQ to allow TotalView for $1.99/month instead of paying IBKR hundreds in fees per month to get limited depth views.

#### Options

```haskell
TSLA  210716C00700000: [u   653.98 ( -7.04%)] [iv 0.47]  3.37 ( -32.57%   -1.65  5.00) (  14.29%    0.40  2.95) ( -24.24%   -1.10  4.45)   3.35 x      4    3.40 x      3
```

Live option quotes display more details because prices move faster:

- quote symbol (OCC symbol *with* spaces) (`TSLA  210716C00700000`)
- live underlying quote price (`653.98`)
- percentage difference between live underlying price and strike price (`-7.04%`)
    - with current underlying price `653.98` and strike at `$700.00`, the stock is currently 7.04% under ATM for the strike:
        - `653.98 * 1.0704 = 700.020192`
- iv for contract based on [last traded price as calculated by IBKR](https://interactivebrokers.github.io/tws-api/option_computations.html) (`[iv 0.47]`)
- current estimated price (`3.37`)
- high: percentage, point difference, and traded high for the session (`( -32.57%   -1.65  5.00)`)
    - current option price of `3.37` is `-32.57%` (`-1.65` points down) from high of day, which was `5.00`
    - will always be negative (or 0% if trading at high of day)
- low: percentage, point difference, and traded low for the session (`(  14.29%    0.40  2.95)`)
    - will always be positive (or 0% if trading at low of day)
- close: percentage, point difference, and last traded price from previous session (`( -24.24%   -1.10  4.45)`)
- NBBO bid price x size (`3.35 x      4`)
- NBBO ask price x size (`3.40 x      3`)

For more detailed option level 2 quotes, we'd recommend [futu/moomoo](https://www.moomoo.com/download/appStore) in-app real time OPRA Level 2 quotes for $3.99/month. They also have a built-in unusual options volume scanner, and they now have [a trading and quote API](https://openapi.moomoo.com/futu-api-doc/en/), but [API quotes are billed differently than in-app quotes](https://qtcard.moomoo.com/buy?market_id=2&qtcard_channel=2&good_type=1024#/) and the API docs are [poorly translated into english](https://openapi.moomoo.com/futu-api-doc/en/qa/opend.html#3043), so ymmv.

futu/moomoo also lets you buy equity depth for ARCA/OpenBook/CBOE/BZX/NASDAQ exchanges, but they charge regular market prices for each of those, so you'd be paying over $100/month for full depth coverage (and their TotalView alone is $25.99/month while WeBull offers it for $1.99/month).

#### Quote Order

The quote view uses this order for showing quotes based on security type:

- futures
- stocks / etf / etn / warrants / etc
- single future options
- single options
- option spreads

Futures are sorted first by our specific futures order defined by the `FUT_ORD` dict in `cli.py` then by name if there isn't a specific sort order requested (because we want `/ES` `/NQ` `/YM` first in the futures list and not `/GBP` `/BTC` etc).

Stocks/etfs sort in REVERSE alphabetical order because it's easier to see the entry point of the stocks view at the lowest point rather than visually tracking the break where futures and stocks meet.

Single option quotes are displayed in alphabetical order.

Finally, option spreads show last because they are multi-line displays showing each symbol leg per spread.

The overall sort is controlled via `cli.py:sortQuotes()`

## How to Login

IBKR only exposes their trade API via a gateway application (Gateway or TWS) which proxies requests between your API consumer applications and the IBKR upstream API itself.

First download the IBKR Gateway, login to the gateway (which will manage the connection attempts to IBKR trade and data services), then have your CLI connect to the gateway.

### Download Gateway

- Download the [IBKR Gateway](https://www.interactivebrokers.com/en/index.php?f=16457) (you can also use TWS as an API gateway, but TWS wastes extra resources if you only need API access and also crashes if you have certain OS enhancements running)
- Login to the gateway with your IBKR username and password (will require 2FA to your phone using the IBKR app)
    - The IBKR gateway will disconnect a minimum of twice per day:
        - IBKR gateway insists on restarting itself once per day, but you can modify the daily restart time.
            - this restart is a hard restart where your CLI application will lose connection to the IBKR gateway itself since the gateway process exits and restarts (note: IBKR Gateway caches your login credentials for up to a week, so you usually don't have to re-login or 2fa when it auto-restarts nightly)
        - The gateway will also have a 30 second to 5 minute upstream network disconnect once per night when the remote IBKR systems do a nightly reboot on their own. During this time your CLI application will remain connected to the gateway, but won't be receiving any updates and can't send any new orders or requests. Also this downtime happens while futures markets are open, but you won't be able to access markets during the nightly reboot downtime, so make sure your risk is managed via brackets or stops.
    - There is no reliable way to run the IBKR Gateway completely unattended for multiple weeks due to the manual 2FA and username/password re-entry process.

### Download `icli`

Download `icli` as a new repo:

```bash
git clone https://github.com/mattsta/icli
```

Create your local environment:

```bash
poetry install
```

Even though you are logged in to the gateway, the IBKR API still requires your account ID for some actions (because IBKR allows multiple account management, so even if you are logged in as you, it needs to know which account you _really_ want to modify).

- Configure your IBKR account id as environment variable or in `.env.icli` as:
    - `ICLI_IBKR_ACCOUNT_ID="U..."`

- You can also configure the gateway host and port to connect to using:
    - `ICLI_IBKR_HOST="127.0.0.1"`
    - `ICLI_IBKR_PORT=4001`
        - host and port are configured in the gateway settings and can be different for live and paper trading
        - the gateway defaults to localhost-only binding and read-only mode

- You can also configure the idle refresh time for toolbar quotes (in seconds):
    - `ICLI_REFRESH=3.3`



Configure environment settings as above, confirm the IBKR Gateway is started (and confirm whether you want read-only mode or full mode in addition to noting which port the gateway is opening for local connections), login to the IBKR Gateway (requires 2fa to the IBKR app on your phone), then run:

```bash
ICLI_IBKR_PORT=[gateway localhost port] poetry run icli
```

You should see your account details showing in the large bottom toolbar along with a default set of quotes (assuming you have all the streaming market data permissions required).

View all commands by just hitting enter on a blank line.

View all commands by category by entering `?`.

View per-command documentation by entering a command name followed by a question: `limit?` or `lim?` or `exec?` or `pos?` or `cancel?` etc.

If you have any doubt about how a command may change your account, check the source for the command in `lang.py` yourself just to confirm the data workflow.

## Caveats

- the IBKR API doesn't allow any operations on fractional shares, so those orders must be handled by IBKR web, mobile, or TWS.
- It's best to have *no* standing orders placed from IBKR web, mobile, or TWS. All orders should be placed from the API itself due to how IBKR handles ["order ID binding" issues](https://interactivebrokers.github.io/tws-api/modifying_orders.html).
    - for example, if you have a GTC REL order to SELL 100 AAPL at $200 minimum placed from IBKR mobile/web/tws which you expect to maybe hit in 3 months, connecting to the IBKR API will actually cancel and re-submit the live order every time you start your API client (instead of the expected behavior of only submitting once daily when the market opens).
    - So, it's best to only place orders through the API endpoints if you are doing majority API-related trading.
    - Also, you can always modify and cancel orders placed via API using the regular mobile/web/tws apps too.
- IBKR only allows one login per account across all platforms, so when your IBKR Gateway is running, you can't login to IBKR mobile/web without kicking the API gateway connection offline (so your API client will lose access to the IBKR network too).
    - though, you can run an unlimited number of *local* clients connecting to the gateway. Useful for things like: if you wanted to develop your own quote viewing/graphing system while also using another api application for trading or for creating a live account balance dashboard, etc.
- IBKR supports many currencies and many countries and many exchanges, but currently `icli` uses USD and SMART exchange transactions for all orders (except for futures which use the correct futures exchange per future symbol).
- IBKR paper trading / sandbox interface doesn't support all features of a regular account, so you may get random errors ("invalid account code") you won't see on your live account. Also you'll probably be unable to use many built-in algo types with paper trading. The only safe paper trading order types appear to be direct limit and market orders.

You should also be comfortable diving into the code if anything looks wonky to you.

## System Limits

`icli` is still limited by all regular IBKR policies including, but not limited to:

- by default the gateway is in Read Only mode
    - Configure -> Settings -> API -> Settings -> 'Read-Only API'
- IBKR has no concept of "buy to open" / "sell to open" / "sell to close" / "buy to close" — IBKR only sees BUY and SELL transactions. If you try to sell something you don't own, IBKR will execute the transaction as a new short position. If you try to sell *more* than something you own, IBKR will sell your position then also create a new short position. It's up to you to track whether you are buying and selling the quantities you expect.
- You should likely convert your account to the [PRO Tiered Commission Plan](https://www.interactivebrokers.com/en/index.php?f=1590) so you can receive transaction rebates, lowest margin fees, access to full 4am to 8pm trading hours, and hopefully receive the highest rebates and lowest commissions on the platform.
- you need to [pay for live market data](https://www.interactivebrokers.com/en/index.php?f=14193) to receive live streaming quotes
    - at a minimum you will want "US Securities Snapshot and Futures Value Bundle" and "US Equity and Options Add-On Streaming Bundle" plus "OPRA Top of Book (L1)" for options pricing plus "US Futures Value PLUS Bundle" for non-delayed futures quotes (doesn't include VIX though)
- you are still limited to "[market data lines](https://www.interactivebrokers.com/en/index.php?f=14193#market-data-display)" where you are limited to 100 concurrent streaming quotes unless you either have a *uuuuuge* account balance, or unless you pay an additional $30 per month for +100 more top of book streaming quotes (but you can buy up to 10 quote booster ultra rare hologram packs, so the max limit on quotes is 100 * 10 + 100 = 1,100 symbols which would also give you access to open 11 concurrent DOM views too)
- you need to manually request access to trade anything not a stock or etf by default (penny stocks, options, bonds, futures, volatility products)
- you need to self-monitor your "[order efficiency ratio](https://ibkr.info/article/1343)" which means you need at least 1 executed trade for every 20 order create/modify/cancel requests you submit to IBKR (you always start the day with 20 free order credits)
    - you can count your daily executed trades using the `executions` command (each row means +20 order credits for the day)
    - there's no built-in mechanism to count how many cancel/modify operations happened in a day because IBKR makes those history rows vanish immediately when processed (we *could* create a counter to log them locally, but haven't had a reason to do so yet)
- IBKR has a fixed limit of 10,000 simultaneous active orders per account (which is a lot tbh)
- typical [FINRA legal restrictions](https://www.finra.org/investors/learn-to-invest/advanced-investing/day-trading-margin-requirements-know-rules) apply such as:
    - if your account is under $25k, for equity symbols and equity options, you are limited to 3 same-day open/close trades per 5 trading days.
        - your live day trades remaining count is visible in the cli toolbar
            - when displayed, the day trades remaining count is updated in real time
            - if your account is over $25k, the count will not display
                - but, if your account has total value bouncing between say $24k and $26k, the limit will appear and vanish and appear again as your balance grows and shrinks above and below the $25k limit.
        - you can hack around the "same-day open/close" limit somewhat with options by turning a winning single-leg option position into a butterfly then closing it all the next day.
    - the $25k 3-per-5 restriction does not apply to futures or future options, so go wild and open then close 1 `/MES` 100 times a night on your $4k account (though, watch out for the $0.52 commission per trade).
    - unlike other brokers, IBKR gives you full 4x day trade margin regardless of your account balance (because IBKR doesn't issue margin calls—their automated systems will try their best (assuming [oil doesn't go negative](https://www.financemagnates.com/forex/brokers/interactive-brokers-loss-from-oil-collapse-swelled-to-104-million/)) to liquidate your positions until you are margin compliant again). so, if your account has $8k equity, you can hold up to $32k of stock during the day (which must be reduced below overnight margin before close—and you are still limited to the 3-in-5 same-day open/close equity trading rules even if closing your 4x margin orders would create a 3-in-5 violation)
- the CBOE options [390 rule](https://support.tastyworks.com/support/solutions/articles/43000435379-what-is-the-390-professional-orders-rule-390-rule-) always applies
    - basically, if you average more than 1 CBOE option order placed every minute for a month (the "390" rule is from 390 minutes being the 6.5 hour trading day; also the order doesn't have to execute, just be placed to count), your account will be [re-classified and everything will cost more for you](https://ibkr.info/node/1242) and you'll potentially [get worse executions](https://markets.cboe.com/us/equities/trading/offerings/retail_priority/) going forward.
- if you aren't deploying an aggressive temporal alpha thesis (i.e. st0nks go brrrr), your orders should be adjusted to not hit a bid/ask price exactly when submitted. Immediate execution is called a "[marketable order](https://ibkr.info/article/201)," and those get the worst commission (more aggressive == more expensive to execute). You can avoiding hitting waiting orders at exchanges manually by adjusting your price (bids lower or asks higher or target a wide midpoint) or you can use various IBKR algo order types which may prefer to not take liquidity immediately (and some IBKR algos you can command to *never* take liquidity for the best rebate probability).
    - restated: you get the best commission rates (and sometimes rebates 😎) when your order is sitting on an exchange's limit order book then *somebody else's* order matches against your waiting order (meaning: the counterparty is being aggressive while you are providing passive liquidity to the market—but if you need to be aggressive, TAKE THE PRICE AND USE IT.)
- also please remember to not run out of money



## Architecture

Entry point for the CLI is [`__main__.py`](icli/__main__.py) which handles the event loop setup, environment variable reading, and app launching.

The easiest way to launch the cli is via poetry in the repository directory: `poetry run icli`

cli commands are processed in a prompt-toolkit loop managed by the somewhat too long `dorepl()` method of class `IBKRCmdlineApp` in [`cli.py`](icli/cli.py).

cli commands are implemented in [`lang.py`](icli/lang.py) with each command being a class with custom argument definitions as organized by the [`mutil/dispatch.py`](https://github.com/mattsta/mutil/blob/main/mutil/dispatch.py) system.  Check out the `OP_MAP` variable for how command names are mapped to categories and implementation classes.

Your CLI session history is persisted in `~/.tplatcli_ibkr_history.{live,sandbox}` so you have search and up/down recall across sessions.

All actions taken by [the underlying IBKR API wrapper](https://github.com/erdewit/ib_insync) are logged in a file named `icli-{timestamp}.log` so you can always review every action the API received (which will also be the log where you can view any series of order updates/modifications/cancels since IBKR removes all intermediate order states of orders after an order is complete).

All times in the interface are normalized to display in US Eastern Time where pre-market hours are 0400-0928, market hours are 0930-1600, and after hours is 1600-2000 (with options trading 0930-1600, with certain etf and index options trading until 1615 every day). Futures operate under their [own weird futures hours](https://www.cmegroup.com/trading-hours.html) schedule, so enjoy trading your Wheat Options and [Mini-sized Wheat Futures](https://www.cmegroup.com/markets/agriculture/grains/mini-sized-wheat.html) between 1900-0745 and 0830-1345.



### Notable Helpers

[`futsexchanges.py`](icli/futsexchanges.py) contains a mostly auto-generated mapping of future symbols to their matching exchanges and full text descriptions. The mapping can be updated by extracting updated versions of [the table of IBKR futures](https://www.interactivebrokers.com/en/index.php?f=26662) using `generateFuturesMapping()`. The reason for this mapping is when entering a futures trade order, IBKR requires the exchange name where the futures symbol lives (i.e. there's no SMART futures router and futures only trade on their owning exchange), so we had to create a full lookup table on our own.
    
Also note: some of the future symbol descriptions are manually modified after the automatic mapping download. Read end of the file for notes about which symbols need additional metadata or symbol changes due to conflicting names or multiplier inconsistencies (example: `BRR` bitcoin contract is one symbol, but microbitcoin is 0.1 multiplier, while standard is 5 multiplier, and for some reason IBKR didn't implement the micro as the standard `MBT` symbol, so you have to use it as `BRR` with explicit multiple).

[`orders.py`](icli/orders.py) is a central location for defining IBKR order types and extracting order objects from specified order types using all [20+ poorly documented, conflicting, and multi-purpose optional metadata fields](https://interactivebrokers.github.io/tws-api/classIBApi_1_1Order.html) an order may need.


## TODO

`icli` is still a work in progress and future features may include:

- better handling of spread orders
- better handling of future entry and exit conditions
- improve visual representation of spreads so you can confirm credit vs. debit transactions
- enable quick placement of bracket orders
- extensible auto-trading hooks
- maybe daily performance reports or efficacy reports based on orders placed vs. modified vs. canceled vs. executed
- add hooks to forward trade notifications as mobile push notifications
- maybe add web interface or real time graph interface
- we may convert the environment variable config to command line params eventually (via click or fire)
- allow optional non-guaranteed spreads for larger accounts
    - these let IBKR run each leg of a spread independently, but you may also not get a complete fill on the spread leading to margin compliance problems if your account isn't big enough
- enable setting more complex optional order conditions like don't fill before or after certain timestamps
- adjust sound infrastructure to play different sounds based on win vs loss vs major win vs major loss vs flat trade capital reclamation
- add cli trade pub/sub infrastructure so you could broadcast your trade actions live to other platforms / webpages / visibility outlets
- more features for "hands off" auto trading operations without needing full custom algo modes
- enable full custom algo modes
- tests? would be a major task to actually mock the IBKR API itself to inject and reply to commands for any CI system.

## History

This is the second trade CLI I've written (the first being a [tradier api cli](https://documentation.tradier.com/brokerage-api) which isn't public yet) because I wanted faster access to scalping options during high volatility times where seconds matter for good trade executions, but navigating apps or web page entry flows wasn't being the most efficient way to place orders.

Writing an IBKR seemed a good way to create a rapid order entry system also capable of using the only public simple fee broker which offers complex [exchange-side and simulated order types](https://www.interactivebrokers.com/en/index.php?f=4985) via API like primary peg (relative orders), mini-algos like adaptive limit/market orders, peg to midpoint, snap to market/primary/midpoint, market with protection, etc.

so here we are.

## Contributions

Feel free to open issues to suggest changes or submit your own PR changes or refactor any existing confusing flows into less coupled components.

Immediate next-step areas of interest are:

- increasing the ease of entering complex trade details without sending users into 8 levels of different nested menus to configure all settings correctly
    - a next step could be writing a linear trade language parser to enable copy/paste between apps like: `OTOCO::BUY_100_TSLA210716C00700000_3.33:SELL_ALL_4.44:STOP_ALL_2.99` which would convert the line into a 3-leg bracket order then execute it all it one step.
- adding extensible custom algo hooks (atr chandelier exits, short/fast moving average crossover buy/sell conditions, etc)
- adding a listening port to accept requests from external applications (probably just a websocket server) so they can use a clean `icli` API to bridge the actual IBKR API
- improving documentation / writing tutorials / helping people not lose money
