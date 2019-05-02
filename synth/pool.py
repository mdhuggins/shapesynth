import numpy as np

import time
from multiprocessing import Pipe, Process, Array
import ctypes
from collections import deque

def _worker(pipe, out_array, array_fill_worker):
    """
    Reads new commands from the pipe and writes them to the beginning of the
    given shared memory array. Generates frames using the given function
    array_fill_worker.
    """

    buf = np.frombuffer(out_array.get_obj())
    while True:
        params = pipe.recv()
        if params == 'STOP':
            break
        buf[:] = 0
        new_wave = array_fill_worker(*params)
        np.copyto(buf[:len(new_wave)], new_wave.astype(ctypes.c_double))
        pipe.send((params, len(new_wave)))


class SharedArrayPool(object):
    """
    A class that allows multiple clients to request computed sequences of data in
    parallel using numpy arrays.
    """

    def __init__(self, buffer_dim, array_fill_worker, num_workers=3):
        """
        buffer_dim: the maximum size of any data sequence that can be processed
        array_fill_worker: a function that takes a requester ID and a tuple of
            params (such as those passed to the request() method), and returns
            a numpy array of data
        num_workers: the number of subprocesses to create
        """

        self.num_workers = num_workers
        self.parent_connections = []
        self.workers = []
        self.buffers = []
        for i in range(self.num_workers):
            parent, child = Pipe()
            self.parent_connections.append(parent)
            mem = Array(ctypes.c_double, buffer_dim)

            buf = np.frombuffer(mem.get_obj())
            self.buffers.append(buf)

            proc = Process(target=_worker, args=(child, mem, array_fill_worker))
            self.workers.append(proc)
            proc.start()

        # Maintain a set of workers that aren't currently working, and a queue of waiting items to be processed
        self.open_workers = set(range(self.num_workers))
        self.waiting = deque()
        self.results = {}

    def request(self, requester, params):
        """
        Requests that the worker function be called with the given parameters.

        requester - a unique ID referencing this request
        params - the params to be passed to the second argument of the worker
            function
        """

        worker_idx = next(iter(self.open_workers), None)
        if worker_idx is None:
            self.waiting.append((requester, params))
            return

        self.parent_connections[worker_idx].send((requester, params))
        self.open_workers.remove(worker_idx)

    def _get(self):
        """Polls each worker pipe and returns the first result it finds, if present."""

        for i, conn in enumerate(self.parent_connections):
            if conn.poll():
                params, length = conn.recv()
                buffer_contents = self.buffers[i][:length]
                self.open_workers.add(i)
                return params, buffer_contents

    def get(self, requester):
        """
        Checks if the given requester's wave has been computed, and if so,
        returns the resulting wave.
        """
        if requester in self.results:
            ret = self.results[requester]
            del self.results[requester]
            return ret
        return None

    def stop(self):
        """Terminates all workers and waits until they join."""

        for conn, proc in zip(self.parent_connections, self.workers):
            conn.send('STOP')
            proc.join()

    def on_update(self):
        # See if a result is available
        obj = self._get()
        if obj is not None:
            # Save it
            (requester, params), contents = obj
            self.results[requester] = np.copy(contents)

        # Update the queue if workers are available
        for i in range(min(len(self.waiting), len(self.open_workers))):
            job = self.waiting.popleft()
            self.request(*job)
