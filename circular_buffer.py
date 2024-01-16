from collections import deque

class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.history = []

    def append(self, item):
        """Appends an item to the circular buffer."""
        self.buffer.append(item)
        if(sum(list(self.buffer)) > 0):
            self.history.append(sum(list(self.buffer)))
    
    def count(self, v):
        return list(self.buffer).count(v)
    
    def sum(self):
        return sum(list(self.buffer))
    
    def flush(self):
        self.__init__(self.size)

    def get_all(self):
        """Returns all items in the circular buffer."""
        return list(self.buffer)

    def __repr__(self):
        """String representation of the circular buffer."""
        return self.buffer.__repr__()
