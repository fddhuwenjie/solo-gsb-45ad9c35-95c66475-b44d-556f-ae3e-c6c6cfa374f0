"""Tap-event review holds (刷卡事件复核暂存与释放).

A review hold freezes a batch of one passenger's tap events for manual review:
each event's *full payload* is copied into an immutable snapshot and the event
is marked ``held``, withdrawing it from pairing, billing and pending queries
until the hold is released. The whole batch is rejected (409, no state change)
when any of the events is already referenced by a reconciliation batch, a
clearing batch or a *resolved* fare dispute -- such references book against the
event's trips and cannot be silently un-settled.

State is append-only, following the disputes pattern: the ``event_holds`` row
and its ``event_hold_snapshots`` rows are written once and never updated; the
open/released state and the optimistic version are derived from whether a row
exists in the separate, append-only ``event_hold_releases`` table (version 1 =
held, version 2 = released).

Release re-settles the passenger from the events' original ``seq``/timestamps
and returns the per-trip increase/decrease amounts. A stale expected version
or an event snapshot that no longer matches the live event row rejects the
release and leaves every event ``held``.
"""
from __future__ import annotations

import hashlib
import json

from .audit import canonical_json
from .models import (
    Adjustment,
    EventHoldOut,
    HeldEventSnapshot,
    HoldReleaseResult,
)


def canonical_hold_payload(body) -> dict:
    """Canonical create request: the event-id list is sorted, so semantically
    identical submissions hash identically regardless of array order."""
    return {
        "hold_id": body.hold_id,
        "passenger_id": body.passenger_id,
        "event_ids": sorted(body.event_ids),
        "reason": body.reason,
    }


def canonical_release_payload(body, hold_id: str) -> dict:
    """Canonical release request; ``hold_id`` comes from the path."""
    return {
        "hold_id": hold_id,
        "expected_version": body.expected_version,
    }


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _trip_event_ids(trip_id: str) -> list[str]:
    """Split a trip_id ("<in_event_id>:<out_event_id>") into its component
    event ids. Trip ids are built by the pairing engine as
    ``f"{in_event_id}:{out_event_id}"``; ids themselves never contain ':' in
    practice, but splitting on every ':' keeps the membership check exact
    (an event id is only ever a whole component, never a substring)."""
    return [part for part in trip_id.split(":") if part]


def find_blocking_reference(conn, event_ids: list[str]) -> tuple[str, str, str] | None:
    """Return ``(kind, event_id, detail)`` for the first event (in submission
    order) already referenced by a booked artefact, else None.

    * a reconciliation batch whose frozen results mention a trip made of the
      event (internal trips are frozen verbatim, with no later recompute);
    * a clearing batch whose immutable ledger split a trip made of the event;
    * a *resolved* dispute (an immutable refund exists) on such a trip.

    Open disputes do not block: they carry no booked money and themselves only
    report differences against the live settlement.
    """
    wanted = set(event_ids)

    recon_rows = conn.execute(
        "SELECT batch_id, trip_id FROM recon_results WHERE trip_id IS NOT NULL"
    ).fetchall()
    for r in recon_rows:
        for eid in _trip_event_ids(r["trip_id"]):
            if eid in wanted:
                return (
                    "reconciliation_batch",
                    eid,
                    f"event '{eid}' is part of trip '{r['trip_id']}' frozen by"
                    f" reconciliation batch '{r['batch_id']}'",
                )

    clearing_rows = conn.execute(
        "SELECT clearing_id, trip_id FROM clearing_entries"
    ).fetchall()
    for r in clearing_rows:
        for eid in _trip_event_ids(r["trip_id"]):
            if eid in wanted:
                return (
                    "clearing_batch",
                    eid,
                    f"event '{eid}' is part of trip '{r['trip_id']}' booked by"
                    f" clearing batch '{r['clearing_id']}'",
                )

    refund_rows = conn.execute(
        "SELECT d.dispute_id, d.trip_id FROM refunds r"
        " JOIN disputes d ON d.dispute_id = r.dispute_id"
    ).fetchall()
    for r in refund_rows:
        for eid in _trip_event_ids(r["trip_id"]):
            if eid in wanted:
                return (
                    "resolved_dispute",
                    eid,
                    f"event '{eid}' is part of trip '{r['trip_id']}' covered by"
                    f" resolved dispute '{r['dispute_id']}'",
                )
    return None


def _snapshot_model(row) -> HeldEventSnapshot:
    payload = json.loads(row["payload"])
    return HeldEventSnapshot(
        event_id=row["event_id"],
        passenger_id=payload["passenger_id"],
        station_id=payload["station_id"],
        tap_type=payload["tap_type"],
        occurred_at=payload["occurred_at"],
        seq=payload["seq"],
        prior_status=row["prior_status"],
        pending_reason=payload["pending_reason"],
    )


def _release_model(row) -> HoldReleaseResult:
    stored = json.loads(row["result"])
    return HoldReleaseResult(
        expected_version=row["expected_version"],
        released_at=row["created_at"],
        adjustments=[Adjustment(**a) for a in stored["adjustments"]],
        affected_trip_count=stored["affected_trip_count"],
        net_delta_cents=stored["net_delta_cents"],
    )


def hold_response(conn, hold_row, status: str) -> EventHoldOut:
    """Build the API view for a stored hold. ``status`` is the replay marker
    ("created" or "duplicate"); the persistent state is derived from whether
    the append-only release row exists, never from a mutable column."""
    snap_rows = conn.execute(
        "SELECT * FROM event_hold_snapshots WHERE hold_id = ? ORDER BY seq",
        (hold_row["hold_id"],),
    ).fetchall()
    release_row = conn.execute(
        "SELECT * FROM event_hold_releases WHERE hold_id = ?",
        (hold_row["hold_id"],),
    ).fetchone()
    return EventHoldOut(
        hold_id=hold_row["hold_id"],
        status=status,
        state="released" if release_row is not None else "held",
        version=2 if release_row is not None else 1,
        passenger_id=hold_row["passenger_id"],
        event_ids=json.loads(hold_row["event_ids"]),
        reason=hold_row["reason"],
        created_at=hold_row["created_at"],
        snapshots=[_snapshot_model(r) for r in snap_rows],
        freeze_adjustments=[
            Adjustment(**a) for a in json.loads(hold_row["freeze_adjustments"])
        ],
        release=_release_model(release_row) if release_row is not None else None,
        released_at=release_row["created_at"] if release_row is not None else None,
    )


def snapshot_matches_event(snap_row, event_row) -> bool:
    """A release only proceeds when every frozen payload still equals the live
    event row: someone with direct database access must not be able to swap an
    event's payload out from under the review and have that change booked."""
    payload = json.loads(snap_row["payload"])
    return (
        event_row is not None
        and event_row["passenger_id"] == payload["passenger_id"]
        and event_row["station_id"] == payload["station_id"]
        and event_row["tap_type"] == payload["tap_type"]
        and event_row["occurred_at"] == payload["occurred_at"]
        and event_row["seq"] == payload["seq"]
    )
