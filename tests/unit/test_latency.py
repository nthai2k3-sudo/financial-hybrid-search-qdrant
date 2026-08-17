"""Unit tests for LatencyStats and LatencyTracker."""

from hybrid_search.evaluation.latency import LatencyStats, LatencyTracker


def test_latency_stats_calculation():
    measurements = [10.0, 12.0, 14.0, 16.0, 20.0]
    stats = LatencyStats.calculate(measurements)
    assert stats.count == 5
    assert stats.min_ms == 10.0
    assert stats.max_ms == 20.0
    assert stats.p50_ms == 14.0
    assert stats.mean_ms == 14.4


def test_latency_tracker():
    tracker = LatencyTracker()
    tracker.record("encoding", 5.2)
    tracker.record("encoding", 6.8)
    tracker.record("retrieval", 12.1)

    enc_stats = tracker.get_stats("encoding")
    assert enc_stats.count == 2
    assert enc_stats.mean_ms == 6.0

    ret_stats = tracker.get_stats("retrieval")
    assert ret_stats.count == 1
    assert ret_stats.mean_ms == 12.1
