"""Command: alias

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


@command(names=["alias"])
@dataclass
class IOpAlias(IOp):
    cmd: str = field(init=False)
    args: list[str] = field(init=False)

    def argmap(self):
        return [DArg("cmd"), DArg("*args")]

    async def run(self):
        # TODO: allow aliases to read arguments and do calculations internally
        # TODO: should this just be an external parser language too?
        aliases = {
            "buy-spx": {"async": ["fast spx c :1 0 :2*"]},
            "sell-spx": {"async": ["evict SPXW* -1 0"]},
            "evict": {"async": ["evict * -1 0"]},
            "clear-quotes": {"async": ["qremove blahblah SPXW*"]},
        }

        if self.cmd not in aliases:
            logger.error("[alias {}] Not found?", self.cmd)
            logger.error("Available aliases: {}", sorted(aliases.keys()))
            return None

        cmd = aliases[self.cmd]
        logger.info("[alias {}] Running: {}", self.cmd, cmd)

        # TODO: we could make a simpler run wrapper for "run command string" instead of
        #       always breaking out the command-vs-argument strings manually.
        return await asyncio.gather(
            *[
                self.runoplive(
                    cmd.split()[0],
                    " ".join(cmd.split()[1:]),
                )
                for cmd in cmd["async"]
            ]
        )
