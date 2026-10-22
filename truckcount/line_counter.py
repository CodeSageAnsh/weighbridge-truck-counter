"""
Debounced line-occupancy counter.

A weighbridge holds one truck at a time, so instead of tracking IDs across
frames we treat a short virtual line on the bridge as a sensor with two states:

    CLEAR  --(line covered for >= enter_frames)-->  COVERED   (count += 1)
    COVERED --(line clear  for >= clear_frames)-->  CLEAR

The two thresholds remove detector flicker: a box that blinks out for a few
frames does not create a second count, and a truck reversing past the end of
the line is never counted because the line is clipped to the bridge slab.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Line:
    x: int
    top: int
    bottom: int

    def covered_by(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """True if a bounding box spans the line horizontally and overlaps it vertically."""
        return x1 <= self.x <= x2 and not (y2 < self.top or y1 > self.bottom)


@dataclass
class LineCounter:
    enter_frames: int
    clear_frames: int
    state: str = "CLEAR"
    total: int = 0
    _covered_streak: int = field(default=0, repr=False)
    _clear_streak: int = field(default=0, repr=False)

    def update(self, covered: bool) -> bool:
        """Feed one frame. Returns True on the frame a new vehicle is counted."""
        if covered:
            self._covered_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._covered_streak = 0

        if self.state == "CLEAR" and self._covered_streak >= self.enter_frames:
            self.state = "COVERED"
            self.total += 1
            return True
        if self.state == "COVERED" and self._clear_streak >= self.clear_frames:
            self.state = "CLEAR"
        return False
