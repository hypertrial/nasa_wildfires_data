"""
Parallel Processing Utilities
============================

Utility functions for parallelizing data processing tasks.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial


def get_optimal_workers():
    """
    Determine the optimal number of worker processes based on CPU cores.

    Returns:
        int: Optimal number of worker processes (cores-1, minimum 1)
    """
    # Determine optimal number of workers based on CPU cores
    n_cores = multiprocessing.cpu_count()
    # Use at most cores-1 to leave one core for system processes
    n_workers = max(1, n_cores - 1)

    return n_workers


def process_in_parallel(
    func, item_list, n_workers=None, use_threads=False, *args, **kwargs
):
    """
    Process a list of items in parallel using a specified function.

    Args:
        func: Function to execute on each item
        item_list: List of items to process
        n_workers: Number of worker processes/threads to use.
            If None, uses the optimal number based on CPU cores.
        use_threads: If True, uses ThreadPoolExecutor instead of ProcessPoolExecutor.
            Threads are better for I/O-bound tasks, processes for CPU-bound tasks.
        *args, **kwargs: Additional arguments to pass to the function

    Returns:
        list: Results from processing each item
    """
    if n_workers is None:
        n_workers = get_optimal_workers()

    # Create a partial function with fixed arguments
    if args or kwargs:
        func = partial(func, *args, **kwargs)

    # Use either thread pool or process pool based on the parameter
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    # Process items in parallel
    with executor_cls(max_workers=n_workers) as executor:
        results = list(executor.map(func, item_list))

    return results


def batch_items(items, n_batches=None, batch_size=None):
    """
    Split a list of items into batches for parallel processing.

    Either n_batches or batch_size must be provided.

    Args:
        items: List of items to batch
        n_batches: Number of approximately equal-sized batches to create
        batch_size: Specific size for each batch

    Returns:
        list: List of batches, where each batch is a slice of the input list
    """
    n_items = len(items)

    if batch_size is not None:
        # Create batches of the specified size
        return [items[i : i + batch_size] for i in range(0, n_items, batch_size)]
    elif n_batches is not None:
        # Create approximately equal-sized batches
        batch_size = max(1, n_items // n_batches)
        return [items[i : i + batch_size] for i in range(0, n_items, batch_size)]
    else:
        # If neither is provided, create one batch per worker
        n_workers = get_optimal_workers()
        return batch_items(items, n_batches=n_workers)


def create_node_batches(node_ids, node_coords, n_workers):
    """
    Create batches of nodes for parallel processing.

    Args:
        node_ids: List of node IDs
        node_coords: Array of node coordinates
        n_workers: Number of worker processes

    Returns:
        list: List of (start_idx, end_idx, node_ids_slice) tuples for each batch
    """
    n_nodes = len(node_ids)
    batch_size = max(1, n_nodes // n_workers)

    node_batches = [
        (i, min(i + batch_size, n_nodes), node_ids[i : min(i + batch_size, n_nodes)])
        for i in range(0, n_nodes, batch_size)
    ]

    return node_batches
