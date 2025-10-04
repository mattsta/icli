"""Command: expand

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import asyncio
import itertools

import mutil.expand


@command(names=["expand"])
@dataclass
class IOpExpand(IOp):
    """Schedule multiple commands to run based on the expansion of all inputs.

    Example:

    > expand buy {AAPL,MSFT} $10_000 AF

    Would run 2 async commands:
    > buy AAPL $10_000 AF
    > buy MSFT $10_000 AF

    Or you could even do weird things like:

    > expand buy {NVDA,AMD} {$5_000,$9_000} {AF, LIM}

    Would run all of these:
    > buy NVDA $5_000 AF
    > buy NVDA $5_000 LIM
    > buy NVDA $9_000 AF
    > buy NVDA $9_000 LIM
    > buy AMD $5_000 AF
    > buy AMD $5_000 LIM
    > buy AMD $9_000 AF
    > buy AMD $9_000 LIM

    Note: Using 'expand' for menu based commands like "limit" or "spread" will probably do weird/bad things to your interface.
    """

    parts: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*parts", desc="Command to expand then execute all combinations thereof"
            )
        ]

    async def run(self):
        # build each part needing expansion
        logger.info("Expanding request into commands: {}", " ".join(self.parts))

        assemble = [
            mutil.expand.expand_string_curly_braces(part) for part in self.parts
        ]

        # now generate all combinations of all expanded parts
        # (we break out the solution as list of [command, args] pairs)
        cmds = [(x[0], " ".join(x[1:])) for x in itertools.product(*assemble)]
        logger.info(
            "Running commands ({}): {}", len(cmds), [c[0] + " " + c[1] for c in cmds]
        )

        # now run all commands concurrently(ish) by using our standard [command, args] format to the op runner
        try:
            return await asyncio.gather(*[self.runoplive(c[0], c[1]) for c in cmds])
        except:
            logger.exception("Exception in multi-command execution?")
            return None
