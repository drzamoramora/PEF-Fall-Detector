"""Tests for alert construction and dispatch (§3.5 output, §3.6 transports).

The alert is the system's only externally visible output, so what is pinned
here is what a caregiver would receive: that it fires on the right verdicts
and stays silent on the others, that it carries the evidence rather than a
bare "FALL DETECTED", and that one broken transport cannot silence the rest.

Run with:  python -m unittest discover tests
"""

from __future__ import annotations

import unittest

from pef_fall_detector.alerts import Alert, AlertDispatcher


def alert(verdict: str = "stage3_confirmed", severity: str = "severe",
          **kw) -> Alert:
    kw.setdefault("timestamp", 12.5)
    kw.setdefault("frame_index", 375)
    return Alert(verdict=verdict, severity=severity, **kw)


class TestWhatGetsDispatched(unittest.TestCase):

    def test_a_confirmed_fall_alerts(self) -> None:
        d = AlertDispatcher()
        self.assertTrue(d.dispatch(alert()))
        self.assertEqual(len(d.dispatched), 1)

    def test_a_nullified_event_does_not_alert(self) -> None:
        # The subject got up. §3.5: "the alarm is nullified". Alerting anyway
        # is how a fall detector trains its users to ignore it.
        d = AlertDispatcher()
        self.assertFalse(d.dispatch(alert("stage3_nullified", "mild")))
        self.assertEqual(d.dispatched, [])

    def test_a_geometrically_rejected_event_does_not_alert(self) -> None:
        d = AlertDispatcher()
        self.assertFalse(d.dispatch(alert("stage2_rejected", "")))

    def test_unresolved_events_do_not_alert_by_default(self) -> None:
        # Recorded, not dispatched — a deliberate reading of §3.5, and the
        # reason the verdict list is configurable rather than hard-coded.
        d = AlertDispatcher()
        self.assertFalse(d.dispatch(alert("stage3_unresolved", "moderate")))

    def test_the_dispatch_policy_is_configurable(self) -> None:
        d = AlertDispatcher(dispatch_verdicts=("stage3_confirmed",
                                               "stage3_unresolved"))
        self.assertTrue(d.dispatch(alert("stage3_unresolved", "moderate")))


class TestSinks(unittest.TestCase):

    def test_every_sink_receives_the_alert(self) -> None:
        seen_a, seen_b = [], []
        d = AlertDispatcher([seen_a.append, seen_b.append])
        d.dispatch(alert())
        self.assertEqual(len(seen_a), 1)
        self.assertEqual(len(seen_b), 1)

    def test_a_failing_sink_does_not_silence_the_others(self) -> None:
        # The property that matters on a safety device: a dead SMS gateway
        # must not swallow the local buzzer.
        def broken(_: Alert) -> None:
            raise RuntimeError("gateway down")

        survived = []
        d = AlertDispatcher([broken, survived.append])
        self.assertTrue(d.dispatch(alert()))
        self.assertEqual(len(survived), 1)
        self.assertEqual(len(d.failures), 1)

    def test_a_dispatcher_with_no_sinks_still_records(self) -> None:
        # Before Phase 7 there is no transport at all; the record must still
        # show that an alert was warranted.
        d = AlertDispatcher()
        self.assertTrue(d.dispatch(alert()))
        self.assertEqual(len(d.dispatched), 1)


class TestMessage(unittest.TestCase):
    """§3.5's explainability claim, at the moment it matters most."""

    def test_the_message_carries_the_quantities(self) -> None:
        msg = alert(t_deg=82.4, v_tps=-4.1, p_outside_fraction=0.75,
                    max_immobility_s=6.2, prior_state="WALKING").message()
        for fragment in ("SEVERE", "82.4", "-4.10", "75%", "6.2", "WALKING"):
            self.assertIn(fragment, msg, f"falta {fragment!r} en: {msg}")

    def test_unmeasurable_quantities_read_as_na_not_zero(self) -> None:
        # A missing P must not appear as "COM outside 0% of window", which
        # reads as evidence against a fall that nobody actually gathered.
        msg = alert(t_deg=82.4, v_tps=-4.1).message()
        self.assertIn("n/a", msg)
        self.assertNotIn("0% of window", msg)

    def test_the_severity_tag_is_visible_first(self) -> None:
        # The recipient decides whether to run. The tag leads the line.
        self.assertTrue(alert(severity="mild").message().startswith("FALL [MILD]"))


if __name__ == "__main__":
    unittest.main()
