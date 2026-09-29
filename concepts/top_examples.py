from collections import defaultdict
import heapq
import random


class TopExampleStore:
    def __init__(self, top_n: int, random_n: int):
        self.top_n = top_n
        self.random_n = random_n
        self.top = defaultdict(list)
        self.counts = defaultdict(int)
        self.sums = defaultdict(float)
        self.max_values = defaultdict(float)
        self.random_examples = []
        self.random_seen = 0
        self.serial = 0

    def add_random(self, payload: dict) -> None:
        self.random_seen += 1
        if len(self.random_examples) < self.random_n:
            self.random_examples.append(payload)
        else:
            j = random.randint(0, self.random_seen - 1)
            if j < self.random_n:
                self.random_examples[j] = payload

    def update(self, expert_id: int, value: float, payload: dict) -> None:
        self.counts[expert_id] += 1
        self.sums[expert_id] += float(value)
        self.max_values[expert_id] = max(self.max_values[expert_id], float(value))
        self.serial += 1
        item = (float(value), self.serial, payload)
        heap = self.top[expert_id]
        if len(heap) < self.top_n:
            heapq.heappush(heap, item)
        else:
            heapq.heappushpop(heap, item)

    def expert_ids(self) -> list[int]:
        return sorted(self.top.keys())

    def top_examples(self, expert_id: int) -> list[dict]:
        items = sorted(self.top[expert_id], key=lambda x: x[0], reverse=True)
        rows = []
        for value, _, payload in items:
            row = dict(payload)
            row["activation"] = value
            rows.append(row)
        return rows

    def stats(self, expert_id: int) -> dict:
        count = self.counts[expert_id]
        mean = self.sums[expert_id] / max(count, 1)
        return {"active_count": count, "mean_active_activation": mean, "max_activation": self.max_values[expert_id]}
