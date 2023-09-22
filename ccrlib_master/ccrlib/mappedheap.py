import sys
from itertools import islice


def _parent(i): return (i - 1) >> 1


def _left(i): return (i << 1) + 1


def _right(i): return (i << 1) + 2


def _identity(x):
    return x


class MappedHeap:

    def __init__(self, array, length, get_value=_identity):
        self.a = array
        self.get_value = get_value
        self.sorted_size = 0
        self.heap_size = 0
        self.keyToIndex = dict()
        self.build_heap(length)

    def capacity(self):
        return len(self.a)

    def size(self):
        return self.heap_size

    def sorted_heap_size(self):
        return self.sorted_size

    def exchange(self, i, j):
        self.a[i], self.a[j] = self.a[j], self.a[i]
        self.keyToIndex[self.a[i]] = i
        self.keyToIndex[self.a[j]] = j

    def heapify(self, i):
        a = self.a
        f = self.get_value
        l = _left(i)
        r = _right(i)
        largest = l if l < self.heap_size and f(a[l]) > f(a[i]) else i
        largest = r if r < self.heap_size and f(a[r]) > f(a[largest]) else largest
        if largest != i:
            self.exchange(i, largest)
            self.heapify(largest)

    def heapify_key(self, key):
        self.heapify(self.keyToIndex[key])

    def build_heap(self, length):
        self.sorted_size = 0
        self.heap_size = length
        self.keyToIndex.clear()
        for i, k in enumerate(islice(self.a, self.heap_size)):
            self.keyToIndex[k] = i
        for i in range(self.heap_size - 1, -1, -1):
            self.heapify(i)

    def sort_heap(self):
        self.sorted_size = self.heap_size
        for i in range(self.heap_size - 1, 0, -1):
            self.exchange(i, 0)
            self.heap_size -= 1
            self.heapify(0)

    def heap_sort(self, length):
        self.build_heap(length)
        self.sort_heap()

    def maximum(self):
        return self.a[0]

    def extract_max(self):
        if self.heap_size < 1:
            raise Exception("heap underflow")
        maxi = self.a[0]
        self.a[0] = self.a[self.heap_size - 1]
        self.keyToIndex[self.a[0]] = 0
        del self.keyToIndex[maxi]
        self.heap_size -= 1
        self.heapify(0)
        return maxi

    def extract_key(self, key):
        return self.extract_index(self.keyToIndex[key])

    def extract_index(self, i):
        oldkey = self.a[i]
        self.heap_size -= 1
        del self.keyToIndex[oldkey]
        if 1 <= self.heap_size != i:
            self.a[i] = self.a[self.heap_size]
            self.keyToIndex[self.a[i]] = i
            self.propagate_key_index(i)
        return oldkey

    def propagate_key_index(self, idx):
        a = self.a
        f = self.get_value
        i = idx
        key = a[i]
        p = _parent(i)
        while i > 0 and f(a[p]) < f(key):
            self.exchange(i, p)
            i = p
            p = _parent(p)
        if i == idx:
            self.heapify(i)

    def propagate_key(self, key):
        self.propagate_key_index(self.keyToIndex[key])

    def modify_key_index(self, i, key):
        del self.keyToIndex[self.a[i]]
        self.a[i] = key
        self.keyToIndex[key] = i
        self.propagate_key_index(i)

    def modify_key(self, oldkey, newkey):
        self.modify_key_index(self.keyToIndex[oldkey], newkey)

    def insert(self, key):
        i = self.heap_size
        self.a[i] = key
        self.keyToIndex[key] = i
        self.heap_size += 1
        self.propagate_key_index(i)

    def try_insert(self, key):
        if self.heap_size < self.capacity():
            self.insert(key)
        elif self.get_value(self.a[0]) > self.get_value(key):
            del self.keyToIndex[self.a[0]]
            self.a[0] = key
            self.keyToIndex[key] = 0
            self.heapify(0)

    def get_heap(self):
        return self.a[:self.heap_size]

    def get_sorted(self):
        return self.a[:self.sorted_size]

    def get_array(self):
        return self.a[:]

    def __str__(self):
        sb = []
        if self.heap_size > 0:
            lb = 1
            for i in range(0, self.heap_size):
                if i == 0:
                    sb.append('[')
                else:
                    sb.append(",")
                    if i == lb:
                        sb.append("\n")
                        lb = (lb << 1) + 1
                    sb.append(' ')
                sb.append(str(self.a[i]))
            sb.append("]")
            return "".join(sb)
        else:
            return "[" + ", ".join(self.get_sorted()) + "]"

    def has_heap_property_idx(self, i):
        l = _left(i)
        r = _right(i)
        hp = ((l >= self.heap_size or self.get_value(self.a[l]) <= self.get_value(self.a[i]))
              and (r >= self.heap_size or self.get_value(self.a[r]) <= self.get_value(self.a[i])))
        if not hp:
            print(f"hp lost for $i l=$l r=$r", file=sys.stderr)
        return hp

    def has_heap_property(self):
        return all(self.has_heap_property_idx(i) for i in range(_parent(self.heap_size - 1), -1, -1))

    def has_consistent_mapping(self):
        if self.heap_size != len(self.keyToIndex):
            return False
        return all(idx < self.heap_size and key == self.a[idx] for key, idx in self.keyToIndex.items())


if __name__ == "__main__":
    import random

    fcts = {"identity": _identity, "-x": lambda x: x, "(x-50)**2": lambda x: (x - 50) ** 2}
    for f in fcts:
        a_list = list(range(100))
        random.shuffle(a_list)
        print("\n\nSorting function:",f)
        mh = MappedHeap(a_list, len(a_list), fcts[f])
        print("heap size:",mh.heap_size)
        print(mh)
        print("Heap property:", mh.has_heap_property())
        ct = 0
        while mh.heap_size:
            print(f"{mh.extract_max():2}", end=" ")
            ct += 1
            if ct >= 25:
                print()
                ct = 0
        print()
