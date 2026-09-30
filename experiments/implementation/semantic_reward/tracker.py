"""Deterministic, once-only ordered event tracking."""


class EventTracker:
    def __init__(self, events, budget):
        self.events = list(events)
        self.budget = float(budget)
        self.index = 0

    @property
    def pending(self):
        return self.events[self.index] if self.index < len(self.events) else None

    def update(self, status):
        if self.pending is None or status != "completed":
            return 0.0
        self.index += 1
        return self.budget / len(self.events)

    @property
    def progress(self):
        return self.index / len(self.events) if self.events else 0.0
