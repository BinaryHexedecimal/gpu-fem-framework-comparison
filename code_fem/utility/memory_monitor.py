import subprocess
import time
import threading

class GPUMemoryMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False
        self.max_mem = 0

    def _get_memory(self):
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        return int(result.decode().split("\n")[0])

    def _monitor(self):
        while self.running:
            mem = self._get_memory()
            self.max_mem = max(self.max_mem, mem)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        return self.max_mem