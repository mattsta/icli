"""Command: balance

Category: Portfolio
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import prettyprinter as pp  # type: ignore


@command(names=["balance"])
@dataclass
class IOpBalance(IOp):
    """Return the currently cached account balance summary."""

    fields: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*fields")]

    async def run(self):
        ords = self.state.summary

        # if specific fields requested, compare by case insensitive prefix
        # then only output matching fields
        if self.fields:
            send = {}
            for k, v in ords.items():
                for field in self.fields:
                    if k.lower().startswith(field.lower()):
                        send[k] = v
            logger.info("{}", pp.pformat(send))
        else:
            # else, no individual fields requested, so output the entire summary
            logger.info("{}", pp.pformat(ords))
