#!/usr/bin/env python3

import asyncio
import os
import sys

import icli.cli as cli

from dotenv import load_dotenv
from prompt_toolkit.patch_stdout import patch_stdout
from loguru import logger

# just load our dot files into the environment too
load_dotenv(".env.icli")

# Use more efficient coroutine logic if available
# https://docs.python.org/3.12/library/asyncio-task.html#asyncio.eager_task_factory
if sys.version_info >= (3, 12):
    asyncio.get_event_loop().set_task_factory(asyncio.eager_task_factory)

CONFIG_DEFAULT = dict(
    ICLI_IBKR_HOST="127.0.0.1", ICLI_IBKR_PORT=4001, ICLI_REFRESH=3.33
)

# populate config with defaults if they aren't in the environment
CONFIG = {**CONFIG_DEFAULT, **os.environ}

try:
    ACCOUNT_ID: str = CONFIG["ICLI_IBKR_ACCOUNT_ID"]  # type: ignore
except:
    logger.error(
        "Sorry, please provide your IBKR Account ID [U...] in ICLI_IBKR_ACCOUNT_ID"
    )
    sys.exit(0)

HOST: str = CONFIG["ICLI_IBKR_HOST"]  # type: ignore
PORT = int(CONFIG["ICLI_IBKR_PORT"])  # type: ignore
REFRESH = float(CONFIG["ICLI_REFRESH"])  # type: ignore


async def initcli():
    app = cli.IBKRCmdlineApp(
        accountId=ACCOUNT_ID, toolbarUpdateInterval=REFRESH, host=HOST, port=PORT
    )

    await app.setup()

    if sys.stdin.isatty():
        # patch entire application with prompt-toolkit-compatible stdout
        with patch_stdout(raw=True):
            try:
                await app.runall()
            except (SystemExit, EOFError):
                # known-good exit condition
                pass
            except:
                logger.exception("Major uncaught exception?")
    else:
        logger.error("Attached input isn't a console, so we can't do anything!")

    app.stop()


def runit():
    """Entry point for icli script and __main__ for entire package."""
    try:
        asyncio.run(initcli())
    except (KeyboardInterrupt, SystemExit):
        # known-good exit condition
        ...
    except:
        logger.exception("bad bad so bad bad")


if __name__ == "__main__":
    runit()
