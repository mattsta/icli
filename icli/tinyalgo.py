"""tinalgo is just helpers copy/pasted from mmt because I don't want to add mmt as a dependency"""

from dataclasses import dataclass, field
from collections import deque


@dataclass(slots=True)
class ATR:
    # decay/lookback length
    length: int = 20
    prevClose: float = 0.0
    current: float = 0
    updated: int = 0

    def __post_init__(self) -> None:
        assert self.length >= 1

    def update(self, high: float, low: float, close: float) -> float:
        """Update rolling ATR using O(1) storage"""

        self.updated += 1

        if not self.current:
            # if we have no current history, we start from the local price data,
            # otherwise, if we didn't have this case, we would generate a too#
            # wide range and converge it from the top (where as here, we basically
            # use a too low range and converge it from the lower bound instead).

            # NOTE: if the first update has high==low then
            #       the ATR is 0 because the "range" of two equal prices is 0.
            self.current = high - low  # / self.length
        else:
            # else, if we DO have history, then we use a modified update length
            # to improve the bootstrap process (so we don't have to wait for tiny
            # numbers to converge into the full length, we just mock a dynamic length
            # until we reach  max capacity)
            useUpdateLen = min(self.updated, self.length)

            currentTR = max(
                max(high - low, abs(high - self.prevClose)),
                abs(low - self.prevClose),
            )

            # this is an embedded RMA of the TR to create an ATR
            self.current = (
                currentTR + (useUpdateLen - 1) * self.current
            ) / useUpdateLen

        self.prevClose = close

        return self.current


@dataclass(slots=True)
class ATRLive:
    """An ATR but with live trades instead of bars, so it keeps a tiny local history."""

    length: int = 20
    bufferLength: int = 55
    buffer: deque[float] = field(init=False)
    atr: ATR = field(init=False)

    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.bufferLength)
        self.atr = ATR(self.length)

    @property
    def current(self) -> float:
        # passthrough...
        return self.atr.current

    def update(self, price) -> float:
        self.buffer.append(price)
        high = max(self.buffer)
        low = min(self.buffer)

        return self.atr.update(high, low, price)
