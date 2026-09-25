"""Alert construction and dispatch (§3.5 output, §3.6 transports).

The funnel of §3.5 ends with *"the alert is dispatched with the appropriate
severity tag"*. §3.6 names what dispatch means on the target device: *"a
lightweight alert dispatcher that supports SMS and Email notifications, and
a local audible buzzer"*. This module is the seam between those two: it
builds the alert and hands it to whatever sinks are attached, without
knowing what a sink does.

**Why a seam and not a buzzer call inside the state machine.** The decision
logic must stay pure and testable — it is the part the paper claims is
auditable, and it has to run identically in PEF-Lab, in batch evaluation
(§3.7) and on the Raspberry Pi. Wiring a transport into it would tie the
verdict to the machine it happens to run on. With the seam, Phase 7 adds
SMS, e-mail and buzzer sinks and changes nothing about how falls are
decided.

**The message carries the evidence, not just the verdict.** §3.5's
explainability claim is that *"a downstream reviewer can reconstruct the
decision by reading the four quantities from the corresponding frames"*. An
alert that says only "FALL DETECTED" throws that away at the exact moment
it matters — when a caregiver has to decide whether to act. So every alert
carries T, V, P and I, the severity, and what the subject was doing before
the event.

**One failing sink must not silence the others.** A dead SMS gateway cannot
be allowed to swallow the buzzer. Dispatch isolates each sink's failure and
reports it rather than propagating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

#: Verdicts that produce an alert by default. §3.5 dispatches a confirmed
#: fall; a nullified one is by definition the alarm being called off. Events
#: that no stage could resolve (``stage3_unresolved``,
#: ``stage2_inconclusive``) are RECORDED but not dispatched — a deliberate
#: reading of §3.5, and one worth revisiting with data: on a device meant to
#: catch falls, "I could not tell" may deserve a quieter notification rather
#: than silence. Configurable for exactly that reason.
DEFAULT_DISPATCH_VERDICTS = ("stage3_confirmed",)


@dataclass
class Alert:
    """One dispatched alert, with the evidence behind it."""

    timestamp: float
    severity: str
    verdict: str
    frame_index: int
    #: The four §3.4 quantities as they stood when the trigger fired (T, V)
    #: or when the funnel resolved (P, I). NaN where not measurable.
    t_deg: float = float("nan")
    v_tps: float = float("nan")
    p_outside_fraction: float = float("nan")
    max_immobility_s: float = float("nan")
    #: What the subject was doing before the event, for the human reading it.
    prior_state: str = ""
    source: str = ""
    #: The rule that decided the severity (TriggerEvent.reason). The
    #: quantities above are the evidence; this is what they were held
    #: against. Without it the recipient sees numbers but not the verdict's
    #: logic, which is the half of explainability an alert can carry.
    reason: str = ""

    def message(self) -> str:
        """One-line, human-readable, and complete enough to act on.

        Deliberately not a bare "FALL DETECTED": the recipient has to decide
        whether to go and look, and the quantities are what make that
        decision possible without replaying the video.
        """
        def num(value: float, fmt: str) -> str:
            return "n/a" if math.isnan(value) else format(value, fmt)

        parts = [f"FALL [{self.severity.upper()}]"]
        if self.source:
            parts.append(f"source={self.source}")
        parts.append(f"t={self.timestamp:.2f}s frame={self.frame_index}")
        if self.prior_state:
            parts.append(f"subject was {self.prior_state} before")
        parts.append(
            f"T={num(self.t_deg, '.1f')} deg, V={num(self.v_tps, '+.2f')} torso/s, "
            f"COM outside {num(100 * self.p_outside_fraction, '.0f')}% of window, "
            f"still {num(self.max_immobility_s, '.1f')}s"
        )
        if self.reason:
            parts.append(f"why: {self.reason}")
        return " | ".join(parts)


class AlertDispatcher:
    """Fans an alert out to every attached sink, isolating failures.

    A sink is any callable taking an :class:`Alert`. Phase 7 attaches the
    §3.6 transports (buzzer, SMS, e-mail); PEF-Lab attaches a screen banner;
    the headless runner attaches a console line. None of them is known here.
    """

    def __init__(self, sinks: list[Callable[[Alert], None]] | None = None,
                 dispatch_verdicts: tuple[str, ...] = DEFAULT_DISPATCH_VERDICTS
                 ) -> None:
        self.sinks: list[Callable[[Alert], None]] = list(sinks or [])
        self.dispatch_verdicts = tuple(dispatch_verdicts)
        #: Alerts actually sent, in order — the record a test or an
        #: evaluation run reads instead of watching a buzzer.
        self.dispatched: list[Alert] = []
        #: (sink repr, exception) for every sink that raised.
        self.failures: list[tuple[str, Exception]] = []

    def add_sink(self, sink: Callable[[Alert], None]) -> None:
        self.sinks.append(sink)

    def should_dispatch(self, verdict: str) -> bool:
        return verdict in self.dispatch_verdicts

    def dispatch(self, alert: Alert) -> bool:
        """Send an alert if its verdict warrants one. Returns whether it was.

        Every sink is attempted even if an earlier one raised: on a safety
        device, a dead network path must not silence the local buzzer.
        """
        if not self.should_dispatch(alert.verdict):
            return False
        self.dispatched.append(alert)
        for sink in self.sinks:
            try:
                sink(alert)
            except Exception as exc:                      # noqa: BLE001
                self.failures.append((repr(sink), exc))
        return True


def console_sink(alert: Alert) -> None:
    """Print the alert. The default transport until §3.6 hardware exists."""
    print(f"[ALERT] {alert.message()}", flush=True)
