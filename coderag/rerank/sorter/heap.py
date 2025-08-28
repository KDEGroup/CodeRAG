import copy
from typing import List, TypeVar, Generic, Optional, Dict, Any

T = TypeVar('T')

class IterativeHeapSorter(Generic[T]):
    """
    An iterative sorter implementing heap sort using an m-ary heap.
    The sorting process is decomposed into a series of operations that select the "best" element 
    from m+1 (or fewer, depending on the number of children) elements.
    This selection operation is performed externally (e.g., via a comparison function).
    It can sort the entire list (resulting in ascending order), or just ensure the top_k largest elements
    are correctly sorted and placed at the end of the list 
    (e.g., items[n-top_k] to items[n-1] contain the top_k largest elements in order).
    Supports the iterator protocol.
    """

    def __init__(
        self,
        items: List[T],
        *,
        m_arity: int,
        top_k: Optional[int] = None,
    ):
        assert isinstance(items, list), "items must be a list"
        assert m_arity >= 2, "m_arity must be at least 2 (for a valid heap structure)"
        
        self._items = items  # Sorts in-place
        self._n = len(items)
        self._m_arity = m_arity

        if top_k is None:
            self._effective_top_k = self._n
        else:
            assert isinstance(top_k, int) and top_k >= 0, "top_k must be a non-negative integer"
            self._effective_top_k = min(top_k, self._n)

        self._awaiting_selection = False
        self._selection_context: Optional[Dict[str, Any]] = None
        self._finished = False

        if self._n == 0 or self._effective_top_k == 0:
            self._finished = True
            return

        # Phase 1: Heapify (Building a Max-Heap)
        # The goal is to make self._items a max-heap.
        self._current_phase = 'heapify'
        
        # Index of the node that is the root of the subtree currently being heapified (sifted down).
        # Starts from the parent of the last element and goes up to 0.
        self._heapify_current_sift_root_idx = (self._n - 1 - 1) // self._m_arity 
        
        # Index of the node currently being processed (compared with its children) in an active sift_down operation.
        # Set to -1 if no sift_down step is active (i.e., current node is fine or became a leaf in context).
        self._sift_down_active_parent_idx = -1 # Will be initialized in the first __next__ if needed
        
        self._current_heap_size = self._n # During heapify, the heap technically includes all elements.

        # Phase 2: Extraction (Sorting phase)
        self._extracted_count = 0 # Number of elements placed in their final sorted positions.

        # Initialize the first active parent for sifting if heapify needs to run.
        if self._heapify_current_sift_root_idx >= 0:
             self._sift_down_active_parent_idx = self._heapify_current_sift_root_idx
        else: # List is too small (0 or 1 element, or all elements are leaves w.r.t m_arity). Heapify is effectively done.
            self._current_phase = 'extract'
            # _sift_down_active_parent_idx remains -1; __next__ will handle extraction setup.

    def _get_child_idx(self, parent_idx: int, child_num: int) -> int:
        """ Helper to get the index of the child_num-th child (0-indexed) of parent_idx. """
        return self._m_arity * parent_idx + 1 + child_num

    def __iter__(self):
        return self

    def __next__(self) -> List[T]:
        if self._finished:
            raise StopIteration
        if self._awaiting_selection:
            raise RuntimeError("Must submit previous selection before requesting the next one.")

        # This loop handles state transitions that don't require external selection
        # (e.g., moving to next heapify node, finishing a phase, or handling leaf nodes in sift_down)
        while not self._finished:
            if self._sift_down_active_parent_idx == -1: # No active sift_down, or previous one completed its path.
                                                      # Time to advance overall state or start a new sift_down.
                if self._current_phase == 'heapify':
                    # The sift_down for self._heapify_current_sift_root_idx (from a previous iteration) is fully done.
                    # Move to the next node upwards for heapify.
                    self._heapify_current_sift_root_idx -= 1
                    if self._heapify_current_sift_root_idx < 0: # Heapify phase complete
                        self._current_phase = 'extract'
                        self._extracted_count = 0
                        self._current_heap_size = self._n # For extraction, heap starts with all N elements
                        # _sift_down_active_parent_idx remains -1, next part of loop handles extraction setup.
                    else: # Continue heapify with the new root for sifting
                        self._sift_down_active_parent_idx = self._heapify_current_sift_root_idx
                
                elif self._current_phase == 'extract':
                    # Sift_down for the previous extraction's root (items[0]) is complete.
                    # Check if enough elements have been extracted (sorted).
                    if self._extracted_count >= self._effective_top_k:
                        self._finished = True
                        raise StopIteration # Should have been caught if loop re-entered, but as safeguard.

                    # Standard Heap Sort: Place current max (items[0]) at its sorted position
                    # by swapping with the last element of the current unsorted heap portion.
                    # Then, shrink the heap.
                    
                    if self._current_heap_size > 0:
                        # The final resting place for the current max element (items[0])
                        # is at the end of the current conceptual heap.
                        final_resting_place_idx = self._current_heap_size - 1
                        
                        if final_resting_place_idx > 0: # Avoid swapping root with itself if it's the only unsorted element
                             self._items[0], self._items[final_resting_place_idx] = \
                                self._items[final_resting_place_idx], self._items[0]
                        # If final_resting_place_idx is 0, items[0] is the last element, already in place.
                    
                    self._current_heap_size -= 1 
                    self._extracted_count += 1

                    if self._extracted_count >= self._effective_top_k or self._current_heap_size <= 0:
                        self._finished = True
                        raise StopIteration # All required elements sorted or heap exhausted.
                    
                    # If heap still has elements to sort and maintain property for.
                    if self._current_heap_size > 1: 
                        self._sift_down_active_parent_idx = 0 # Sift down the new root (items[0])
                    else: 
                        # Heap now has 0 or 1 element. No sifting needed for this extraction step.
                        # _sift_down_active_parent_idx remains -1. Loop will re-evaluate.
                        continue # Re-evaluate overall state in the while loop.
            
            # If we reach here, _sift_down_active_parent_idx should point to the current parent for sifting.
            # Or it might have been changed to -1 by logic above, triggering re-evaluation.
            if self._sift_down_active_parent_idx == -1 : # Ensure we re-loop if state changed to no active sift.
                continue

            current_parent_for_sift = self._sift_down_active_parent_idx
            
            # This check is a safeguard.
            if current_parent_for_sift >= self._current_heap_size :
                self._sift_down_active_parent_idx = -1 # Parent is out of current heap bounds.
                continue 

            elements_to_compare_indices = [current_parent_for_sift]
            elements_to_compare_values = [self._items[current_parent_for_sift]]
            has_children_in_heap = False

            for i in range(self._m_arity):
                child_idx = self._get_child_idx(current_parent_for_sift, i)
                if child_idx < self._current_heap_size: # Child must be within the current heap bounds
                    elements_to_compare_indices.append(child_idx)
                    elements_to_compare_values.append(self._items[child_idx])
                    has_children_in_heap = True
                else:
                    break # No more children for this parent within the heap
            
            if not has_children_in_heap: 
                # Parent is effectively a leaf node for this sift_down step (no children in heap to compare with).
                # So, this particular sift_down path (for current_parent_for_sift) ends here.
                self._sift_down_active_parent_idx = -1 
                continue # Re-evaluate state in the while loop (will go to advance phase/node).

            # If we are here, we have a parent and at least one child to compare.
            # Prepare context for the external selector.
            self._selection_context = {
                'parent_node_idx_in_list': current_parent_for_sift, # original index of parent in self._items
                'all_candidate_indices_in_list': elements_to_compare_indices, # original indices of all m+1 candidates
            }
            self._awaiting_selection = True
            return copy.deepcopy(elements_to_compare_values) # Return copies of values for selection

        # Should only be reached if self._finished became true inside the loop without StopIteration
        raise StopIteration 

    def submit_selection(self, selected_element_relative_idx: int) -> None:
        """
        Submit the relative index of the "best" element selected by the user
        from the list of candidates provided by the last __next__ call.
        """
        if self._finished:
            # Allow submitting even if finished, might be a late submit for the last step.
            # However, no state change should occur if truly finished. Let's make it an error.
            raise RuntimeError("Sorting is already finished. Cannot submit.")
        if not self._awaiting_selection:
            raise RuntimeError("No selection awaiting submission from __next__.")
        
        ctx = self._selection_context
        assert ctx is not None, "Internal error: selection context missing"

        parent_node_original_idx = ctx['parent_node_idx_in_list']
        all_candidate_original_indices = ctx['all_candidate_indices_in_list']

        if not (0 <= selected_element_relative_idx < len(all_candidate_original_indices)):
            raise ValueError(
                f"selected_element_relative_idx {selected_element_relative_idx} is out of bounds "
                f"for candidate list size {len(all_candidate_original_indices)}."
            )

        # This is the index in self._items of the element chosen by the user as "best"
        actual_idx_of_selected_item_in_list = all_candidate_original_indices[selected_element_relative_idx]

        # Max-Heap logic: the "selected" item is assumed to be the largest among those presented.
        # If this largest item is not the parent itself, then it's a child that's larger than the parent.
        if actual_idx_of_selected_item_in_list != parent_node_original_idx:
            # A child was selected as larger. Swap parent and this child.
            # We trust the user selected the actual maximum.
            self._items[parent_node_original_idx], self._items[actual_idx_of_selected_item_in_list] = \
                self._items[actual_idx_of_selected_item_in_list], self._items[parent_node_original_idx]
            
            # Continue sifting down from the position where the original parent element moved to (which is actual_idx_of_selected_item_in_list).
            self._sift_down_active_parent_idx = actual_idx_of_selected_item_in_list
        else:
            # Parent was already the largest (or was selected as such). 
            # This particular sift-down path (for this parent relative to its direct children) terminates here.
            # The overall sift_down for self._heapify_current_sift_root_idx (or for items[0] in extract) might continue
            # if self._sift_down_active_parent_idx was a child of a previous step.
            # Setting to -1 signals this specific parent-children comparison set is resolved.
            self._sift_down_active_parent_idx = -1 

        self._awaiting_selection = False
        self._selection_context = None

    def get_result(self) -> List[T]:
        if not self._finished:
            raise RuntimeError("Sorting not finished yet.")
        # The _items list is sorted in-place.
        # If top_k was used, the largest top_k elements are at the end of the list, sorted ascendingly.
        # e.g., items[n-1] is the largest, items[n-2] is 2nd largest, ..., items[n-top_k] is k-th largest.
        # If top_k == n (full sort), the entire list is sorted ascendingly.
        return self._items
