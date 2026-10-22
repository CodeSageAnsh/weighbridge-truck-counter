import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from truckcount.line_counter import Line, LineCounter


def run(seq, enter=3, clear=4):
    c = LineCounter(enter_frames=enter, clear_frames=clear)
    for covered in seq:
        c.update(bool(covered))
    return c.total


def test_box_must_span_line_and_overlap_slab():
    line = Line(x=596, top=134, bottom=362)
    assert line.covered_by(500, 200, 700, 300)
    assert not line.covered_by(600, 200, 700, 300)   # right of line
    assert not line.covered_by(500, 400, 700, 500)   # below slab (back lane)


def test_one_truck_counts_once():
    assert run([1] * 20) == 1


def test_detector_flicker_does_not_double_count():
    assert run([1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1]) == 1


def test_brief_false_positive_is_ignored():
    assert run([0, 1, 1, 0, 0, 0]) == 0


def test_two_trucks_separated_by_clear_gap():
    assert run([1] * 5 + [0] * 6 + [1] * 5) == 2
