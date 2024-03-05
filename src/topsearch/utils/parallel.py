from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Collection, List


def run_parallel(func: Callable, arglist: Collection, extra_args: List= [], processes: int = 2, return_input = False):

    with ProcessPoolExecutor(processes) as executor:
        futures = {}
        for arg in arglist:
            future = executor.submit(func, arg, *extra_args)
            futures[future] = arg 

        for future in as_completed(futures):
            if return_input:
                yield futures[future], future.result()
            else:     
                yield future.result()
        