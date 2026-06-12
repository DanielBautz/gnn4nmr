"""Process-wide singletons and locks.

The app holds everything in memory: the loaded dataset(s), the active model,
the (single) training job and explanation jobs/cache. Flask runs threaded, so
all mutable state is guarded:

- dataset_lock:   dataset registry mutations (load/select)
- model_lock:     swapping the active model
- inference_lock: EVERY model forward/explanation. PyG's GNNExplainer
                  temporarily installs edge masks on the model's
                  MessagePassing modules, so concurrent forwards through the
                  same model would silently see a masked model.
- train_lock:     training job status reads/writes
- jobs_lock:      explanation job registry
- cache_lock:     explanation result cache
"""
import threading
from collections import OrderedDict

from . import config


class AppState:
    def __init__(self):
        self.device = config.resolve_device()

        self.dataset_lock = threading.Lock()
        self.datasets = {}  # file_name -> entry (see dataset_service.load_dataset)
        self.active_dataset_file = None

        self.model_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.active_model = None  # {"model_id", "sidecar", "adapter", "view"}

        self.train_lock = threading.Lock()
        self.train_job = None  # status dict, see training_service
        self.train_thread = None
        self.train_stop = threading.Event()

        self.jobs_lock = threading.Lock()
        self.explain_jobs = OrderedDict()  # job_id -> {"status", "result", "error", ...}

        self.cache_lock = threading.Lock()
        self.explain_cache = OrderedDict()  # cache_key -> result dict

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def active_dataset(self):
        with self.dataset_lock:
            if self.active_dataset_file is None:
                return None
            return self.datasets.get(self.active_dataset_file)

    def training_active(self):
        with self.train_lock:
            return self.train_job is not None and self.train_job["status"] in ("running", "stopping")

    def cache_get(self, key):
        with self.cache_lock:
            result = self.explain_cache.get(key)
            if result is not None:
                self.explain_cache.move_to_end(key)
            return result

    def cache_put(self, key, result, max_size=256):
        with self.cache_lock:
            self.explain_cache[key] = result
            self.explain_cache.move_to_end(key)
            while len(self.explain_cache) > max_size:
                self.explain_cache.popitem(last=False)

    def cache_clear(self):
        with self.cache_lock:
            self.explain_cache.clear()


STATE = AppState()
