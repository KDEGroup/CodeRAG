from typing import List, TypeVar, Generic, Optional
import copy

T = TypeVar('T')


class WindowedBubbleSorter(Generic[T]):
    """
    A bubble sorter using a fixed-size window and controllable step size.
    Under the constraint that only window_size elements can be sorted at once,
    it guarantees only the first top_k elements are sorted, or the entire list
    if top_k is not specified.
    Supports the iterator protocol and can be used directly in for ... in ... loops.
    Includes validation to raise an error if the next window is requested before
    submitting the previous one during iteration.
    """

    def __init__(
        self,
        items: List[T],
        *,
        window_size: int,
        move_step: int = 1,
        top_k: int | None = None,
    ):
        # Parameter validation
        assert isinstance(items, list), "items must be a list"
        assert window_size > 0, "window_size must be positive"
        assert 1 <= move_step <= window_size, "move_step must be between 1 and window_size"

        self._items = items
        self._n = len(items)
        self._k = window_size
        self._step = move_step
        self._top_k = top_k if top_k is not None else self._n
        assert 1 <= self._top_k <= self._n, "top_k must be between 1 and len(items)"

        # Bubble sort double loop pointers
        self._i = 0            # Confirmed prefix boundary
        self._j = self._n - 1  # End index of the current bubbling window
        self._finished = False
        self._awaiting_submit = False  # Flag indicating whether submission of current window sort result is awaited

    def __iter__(self):
        return self

    def __next__(self) -> List[T]:
        """
        Returns a deep copy of the current window to be sorted; ends iteration when sorting is complete.
        Raises an error if called before submitting the previous window.
        """
        if self._finished:
            raise StopIteration
        if self._awaiting_submit:
            raise RuntimeError("Must submit the previous chunk before requesting the next one.")

        start = max(0, self._j - self._k + 1)
        chunk = copy.deepcopy(self._items[start : self._j + 1])
        self._awaiting_submit = True
        return chunk

    def submit_sorted(self, sorted_chunk: List[T]) -> None:
        """
        Accepts the sorted content of the window, updates the main list,
        then advances the bubble pointers according to move_step.
        Raises an error if there is no window awaiting submission.
        """
        if self._finished:
            raise RuntimeError("Cannot submit: already finished")
        if not self._awaiting_submit:
            raise RuntimeError("No chunk awaiting submission.")

        start = max(0, self._j - self._k + 1)
        expected = self._j - start + 1
        assert len(sorted_chunk) == expected, (
            f"Expected chunk length {expected}, got {len(sorted_chunk)}"
        )

        # Write back the sorted results
        self._items[start : self._j + 1] = sorted_chunk
        self._awaiting_submit = False

        # Advance the bubble pointers
        self._j -= self._step
        if self._j <= self._i:
            self._i += self._step
            if self._i >= self._top_k:
                self._finished = True
            else:
                self._j = self._n - 1

    def get_result(self) -> List[T]:
        """
        Returns the final sorted list after sorting is complete,
        otherwise raises a RuntimeError.
        """
        if not self._finished:
            raise RuntimeError("Sorting not finished yet")
        return self._items
