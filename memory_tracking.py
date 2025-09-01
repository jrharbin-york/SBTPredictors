import gc
import os
import tracemalloc
from abc import ABCMeta, abstractmethod

import psutil
import structlog
from filprofiler.api import profile

log = structlog.get_logger()

class MemoryTracker(metaclass=ABCMeta):
    @abstractmethod
    def start_tracking(self):
        pass

    @abstractmethod
    def end_tracking(self):
        pass

    def with_tracking(self, function_to_track):
        return function_to_track()

class NullMemoryTracker(MemoryTracker):
    def start_tracking(self):
        pass

    def end_tracking(self):
        return 0

class TraceMallocMemoryTracker(MemoryTracker):
    def __init__(self):
        pass

    def start_tracking(self):
        tracemalloc.start()

    def end_tracking(self):
        stats = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    def with_tracking(self, function_to_track):
        return function_to_track()

class PSUtilMemoryTracker(MemoryTracker):
    def __init__(self):
        self.with_gc_disabled = False
        self.process = psutil.Process(os.getpid())


    def clear_gc(self):
        log.info("Running GC to clear memory")
        gc.collect()

    def enable_gc(self):
        log.info("Enabling GC")
        gc.enable()

    def disable_gc(self):
        log.info("Disabling GC")
        gc.disable()

    def start_tracking(self):
        if self.with_gc_disabled:
            self.clear_gc()
            self.disable_gc()
        self.mem_info_before = self.process.memory_info()
        log.info("Starting memory tracking")

    def end_tracking(self):
        self.mem_info_after = self.process.memory_info()
        if self.with_gc_disabled:
            self.clear_gc()
            self.enable_gc()
        log.info("Ending memory tracking")
        return self.mem_info_after.rss - self.mem_info_before.rss

    def with_tracking(self, function_to_track):
        return function_to_track()

class PSUtilMemoryTrackerNoGC(PSUtilMemoryTracker):
    def __init__(self):
        super().__init__()
        self.with_gc_disabled = True

class FilProfilerMemoryTracker(MemoryTracker):
    def __init__(self, filename):
        self.filename = filename

    def start_tracking(self):
        # Redundant, the tracking is done in with_tracking
        pass

    def end_tracking(self):
        # Redundant, the tracking is done in with_tracking
        pass

    def set_filename(self, report_filename):
        self.filename = report_filename

    def with_tracking(self, function_to_track):
        return profile(function_to_track, self.filename)