"""Utils.py

Kinda like helpers, but more generic abstractions of repetitive actions.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

###############################################################################
#                                                                             #
#  Duplicate Message Suppressor                                               #
#                                                                             #
#  A utility to track and manage repetitive log messages                      #
#  Created: March 29, 2025                                                    #
#                                                                             #
###############################################################################


@dataclass(slots=True)
class MessageRecord:
    """Tracks information about a specific message occurrence pattern."""

    first_seen: float
    last_reported: float
    last_updated: float  # Track when this record was last updated
    count: int = 1

    def should_report(self, now: float, interval: float) -> bool:
        """Determine if it's time to report this message again."""
        time_since_last_report = now - self.last_reported
        return time_since_last_report >= interval

    def get_summary(self, message: str) -> str:
        """Generate a summary message with occurrence information."""
        first_seen_str = time.strftime("%H:%M:%S", time.localtime(self.first_seen))
        return f"{message} (repeated {self.count} times since {first_seen_str})"

    def reset_counter(self, now: float) -> None:
        """Reset the counter after reporting."""
        self.last_reported = now
        self.last_updated = now
        self.count = 0

    def increment(self, now: float) -> None:
        """Increment the occurrence counter and update the timestamp."""
        self.count += 1
        self.last_updated = now

    def is_stale(self, now: float, expiry_interval: float) -> bool:
        """Check if this record is stale and can be removed from cache."""
        return now - self.last_updated >= expiry_interval


@dataclass(slots=True)
class DuplicateMessageHandler:
    """Handles message deduplication over time intervals."""

    # How often to report recurring messages (in seconds)
    report_interval: float = 30.0

    # How long to keep message records in cache after last update (default: 2x report_interval)
    # Note: an expire interval of '0' means messages are always deleted after their first display,
    #       effectively disabling the deduplication cache entirely.
    cache_expiry_interval: float | None = None

    # Internal storage for tracking messages
    _message_registry: dict[str, MessageRecord] = field(default_factory=dict)

    def __post_init__(self):
        # If cache_expiry_interval not set, use twice the report_interval
        if self.cache_expiry_interval is None:
            self.cache_expiry_interval = self.report_interval * 2

    def handle_message(
        self,
        message: str,
        key: str | None = None,
        log_func: Callable[[str], None] = print,
    ) -> bool:
        """
        Handle a potentially duplicate message.

        Args:
            message: The message to log
            key: Optional unique key to identify the message (defaults to message itself)
            log_func: Function to call for logging (defaults to print)

        Returns:
            bool: True if the message was logged, False if suppressed
        """
        # Clean stale entries periodically
        self._clean_stale_records()

        # Use the message as the key if none provided
        msg_key = key if key is not None else message
        now = time.time()

        # If this is a new message or one we haven't seen in a while
        if msg_key not in self._message_registry:
            self._message_registry[msg_key] = MessageRecord(
                first_seen=now, last_reported=now, last_updated=now
            )
            # Always log the first occurrence
            log_func(message)
            return True

        # Update existing record
        record = self._message_registry[msg_key]
        record.increment(now)

        # Check if it's time to report again
        if record.should_report(now, self.report_interval):
            # Report with count information
            log_func(record.get_summary(message))

            # Reset the counter and update last reported time
            record.reset_counter(now)
            return True

        return False

    def _clean_stale_records(self) -> None:
        """Remove stale message records from the registry."""
        now = time.time()
        keys_to_remove = [
            key
            for key, record in self._message_registry.items()
            if record.is_stale(now, self.cache_expiry_interval or 0)
        ]

        for key in keys_to_remove:
            del self._message_registry[key]

    def clear(self, key: str | None = None) -> None:
        """
        Clear tracking for a specific message or all messages.

        Args:
            key: Key of message to clear, or None to clear all
        """
        if key is not None and key in self._message_registry:
            del self._message_registry[key]
        elif key is None:
            self._message_registry.clear()
