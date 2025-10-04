"""Command: math

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


@command(names=["math"])
@dataclass
class IOpCalculator(IOp):
    """Just show a calculator!"""

    parts: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*parts", desc="Calculator input")]

    async def run(self):
        cmd = " ".join(self.parts)

        try:
            logger.info("[{}]: {:,.4f}", cmd, self.state.calc.calc(cmd))
        except Exception as e:
            logger.warning("[{}]: calculation error: {}!", cmd, e)
