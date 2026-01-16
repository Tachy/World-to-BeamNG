import time


class StepTimer:
    """Kapselt Schritt-Nummerierung, Logging und Zeitmessung."""

    def __init__(self):
        self.steps = []  # list of dicts: {label, title, start, duration}
        self._counter = 0
        self._global_start = time.time()

    @property
    def current_label(self):
        return self.steps[-1]["label"] if self.steps else None

    def begin(self, title: str):
        now = time.time()
        self._close_open(now)
        self._counter += 1
        label = str(self._counter)
        self.steps.append({"label": label, "title": title, "start": now, "duration": None})

        print(f"\n{'='*60}")
        print(f"[{label}] {title}...")
        print(f"{'='*60}")

    def set_duration(self, duration: float):
        """Setze Dauer manuell (z.B. bei Skips nach begin)."""
        for step in reversed(self.steps):
            if step["duration"] is None:
                step["duration"] = duration
                return

    def report(self):
        """Finalisiere offenen Schritt und drucke eine Zeitübersicht."""
        now = time.time()
        self._close_open(now)
        total_elapsed = now - self._global_start
        print(f"\n{'=' * 60}")
        print(f"ZEITMESSUNG (Gesamtzeit: {total_elapsed:.2f}s / {total_elapsed/60:.1f} min)")
        print(f"{'=' * 60}")
        for step in self.steps:
            step_time = step.get("duration") or 0.0
            percentage = (step_time / total_elapsed) * 100 if total_elapsed else 0
            step_display = f"{step['label']} {step['title']}"
            bar_length = int(percentage / 2)  # 50 chars = 100%
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {step_display:.<35} {step_time:>6.2f}s ({percentage:>5.1f}%) {bar}")
        print(f"{'=' * 60}")

    def results(self):
        """Gibt Liste der Schritte inkl. Dauer zurück."""
        return self.steps

    def _close_open(self, now: float | None = None):
        if now is None:
            now = time.time()
        for step in reversed(self.steps):
            if step["duration"] is None:
                step["duration"] = now - step["start"]
                return step["duration"]
        return 0.0
