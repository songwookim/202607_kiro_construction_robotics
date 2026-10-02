"""Worker-thread action waits with ownership of delayed goal responses."""

import threading


def wait_for_future(future, timeout, operation, *, timeout_message=None):
    """Wake on completion, without adding a polling interval to each RPC."""
    completed = threading.Event()
    future.add_done_callback(lambda _future: completed.set())
    if not completed.wait(timeout):
        raise RuntimeError(timeout_message or f"{operation} timed out after {timeout:.0f} s")
    return future.result()


class ActionCall:
    """Cancel a late accepted goal even after its calling worker has timed out.

    Do not cancel the submission Future: its late response is needed to get
    the server goal handle and send an actual ROS action cancellation.
    """

    def __init__(self, description, on_accepted=None):
        self.description = description
        self.on_accepted = on_accepted
        self.handle = None
        self.cancel_error = None
        self._lock = threading.RLock()
        self._accepted = threading.Event()
        self._finished = threading.Event()
        self._abandoned = False
        self._cancel_requested = False
        self._cancel_sent = False
        self._error = None
        self._result = None

    def send(self, client, goal, feedback_callback=None):
        with self._lock:
            if self._cancel_requested:
                raise RuntimeError(f"{self.description} canceled before submission")
            kwargs = {} if feedback_callback is None else {"feedback_callback": feedback_callback}
            client.send_goal_async(goal, **kwargs).add_done_callback(self._goal_ready)

    def _cancel_handle(self):
        if self.handle is not None and not self._cancel_sent:
            try:
                self.handle.cancel_goal_async()
                self._cancel_sent = True
            except Exception as error:
                self.cancel_error = str(error)

    def cancel(self):
        with self._lock:
            self._cancel_requested = True
            self._cancel_handle()

    def abandon(self):
        with self._lock:
            self._abandoned = True
            self.cancel()

    def _goal_ready(self, future):
        with self._lock:
            try:
                handle = future.result()
                if not handle.accepted:
                    self._error = f"{self.description} goal rejected"
                    self._finished.set()
                    return
                self.handle = handle
                if self._abandoned or self._cancel_requested:
                    self._cancel_handle()
                elif self.on_accepted is not None:
                    self.on_accepted(handle)
                if not self._abandoned:
                    handle.get_result_async().add_done_callback(self._result_ready)
            except Exception as error:
                self._error = str(error)
                self.cancel()
                self._finished.set()
            finally:
                self._accepted.set()

    def _result_ready(self, future):
        with self._lock:
            if self._abandoned:
                return
            try:
                self._result = future.result().result
            except Exception as error:
                self._error = str(error)
            self._finished.set()

    def wait(self, acceptance_timeout, result_timeout):
        if not self._accepted.wait(acceptance_timeout):
            self.abandon()
            raise TimeoutError(f"{self.description} goal response timed out")
        if not self._finished.wait(result_timeout):
            self.abandon()
            raise TimeoutError(f"{self.description} timed out")
        if self._error is not None:
            raise RuntimeError(self._error)
        return self._result
