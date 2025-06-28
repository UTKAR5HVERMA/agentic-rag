import threading

class GeminiKeyManager:
    def __init__(self, key_file="gemini_keys.txt"):
        with open(key_file, "r") as f:
            self.keys = [line.strip() for line in f if line.strip()]
        self.lock = threading.Lock()
        self.index = 0

    def get_key(self):
        with self.lock:
            return self.keys[self.index]

    def rotate_key(self):
        with self.lock:
            self.index = (self.index + 1) % len(self.keys)
            return self.keys[self.index]
