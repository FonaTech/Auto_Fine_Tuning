"""
core/dynamic_dataset.py
Thread-safe growing preference dataset buffer for auto SFT→ORPO/DPO conversion.
Producer (rejection generator thread) calls add().
Consumer (trainer) calls get_batch() or wait_for_batch().
"""

import threading
from typing import List, Dict, Optional, Tuple


class DynamicPreferenceDataset:
    """Thread-safe buffer for dynamically generated preference pairs."""

    def __init__(self, prompts: List[Dict], batch_size: int = 4):
        """
        Args:
            prompts: list of {"prompt": str, "chosen": str} — chosen already set
            batch_size: minimum samples needed before training can consume
        """
        self._lock = threading.Lock()
        self._ready: List[Dict] = []
        self._pending: List[Dict] = list(prompts)
        self._batch_size = batch_size
        self._generation_done = threading.Event()
        self._new_data = threading.Event()
        self._consumed_idx = 0
        self.total = len(prompts)
        self.generated = 0
        self.errors = 0

    @property
    def ready_count(self) -> int:
        with self._lock:
            return len(self._ready)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def is_done(self) -> bool:
        return self._generation_done.is_set()

    def pop_pending(self, n: int = 1) -> List[Dict]:
        """Pop up to n pending items for the generator to process."""
        with self._lock:
            batch = self._pending[:n]
            self._pending = self._pending[n:]
            return batch

    def add(self, item: Dict):
        """Add a completed {prompt, chosen, rejected} item."""
        with self._lock:
            self._ready.append(item)
            self.generated += 1
        self._new_data.set()

    def add_error(self):
        """Record a generation error (skip this sample)."""
        with self._lock:
            self.errors += 1

    def mark_done(self):
        """Signal that all generation is complete."""
        self._generation_done.set()
        self._new_data.set()

    def wait_for_batch(self, timeout: float = 300) -> bool:
        """Block until at least batch_size samples are ready. Returns True if ready."""
        while True:
            with self._lock:
                if len(self._ready) >= self._batch_size:
                    return True
            if self._generation_done.is_set():
                with self._lock:
                    return len(self._ready) >= self._batch_size
            self._new_data.clear()
            if not self._new_data.wait(timeout=timeout):
                return False

    def get_all_ready(self) -> List[Dict]:
        """Get all currently ready samples (non-blocking snapshot)."""
        with self._lock:
            return list(self._ready)

    def get_new_since(self, last_idx: int) -> Tuple[List[Dict], int]:
        """Get samples added since last_idx. Returns (new_samples, new_idx)."""
        with self._lock:
            new = self._ready[last_idx:]
            return new, len(self._ready)

    def peek_ready_count(self) -> int:
        """Non-blocking count of currently ready samples."""
        with self._lock:
            return len(self._ready)

    def consume_ready(self, n: int) -> List[Dict]:
        """Take up to n ready samples and remove them from the buffer.
        This releases memory — the buffer does not retain consumed items.
        """
        with self._lock:
            if n <= 0 or not self._ready:
                return []
            batch = self._ready[:n]
            self._ready = self._ready[n:]
            self._consumed_idx += len(batch)
            return batch

    def get_progress(self) -> Tuple[int, int, int]:
        """Returns (generated, total, errors)."""
        with self._lock:
            return self.generated, self.total, self.errors
