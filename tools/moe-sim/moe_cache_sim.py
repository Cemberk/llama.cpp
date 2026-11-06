#!/usr/bin/env python3
"""Replay MoE routing traces against expert cache policies.

A trace comes from `llama-server --moe-cache-mib N --moe-cache-trace FILE`:
one line per cached layer and ubatch, "layer n_tokens expert expert ...".
The simulator gives every policy the same number of slots (experts the
device holds, shared by all traced layers, as llama-moe-cache does for layers
of one device and layout) and reports how many selected experts each would
have had to upload.

Policies:
  slru    the cache in src/llama-moe-cache.cpp: probation and protected halves
  lru     plain least recently used
  lfu     least frequently used, with counts that decay every step
  warm    slru, preloaded with the experts a profile trace used most, as
          --moe-cache-profile does (up to half the slots)
  next    evicts the expert whose next use is expected furthest away: its
          layer's distance in the pass, plus the tokens its count says it
          sits out (after JITEngine's Flash-Next M2 engine)
  belady  evicts the expert used furthest in the future: the best possible

  python moe_cache_sim.py trace.txt --slots 2048
  python moe_cache_sim.py trace.txt --fraction 0.25 --profile older-trace.txt
"""

from __future__ import annotations

import argparse
import collections
import heapq
import sys
from dataclasses import dataclass, field

Key = tuple  # (layer, expert)


@dataclass
class Step:
    """The experts one ubatch selected in one layer."""

    layer: int
    tokens: int
    experts: list


def read_trace(path: str) -> list:
    steps = []
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            layer, tokens = int(parts[0]), int(parts[1])
            steps.append(Step(layer, tokens, [int(p) for p in parts[2:]]))
    return steps


def unique(step: Step) -> list:
    seen = set()
    keys = []
    for e in step.experts:
        if e not in seen:
            seen.add(e)
            keys.append((step.layer, e))
    return keys


class Policy:
    name = ""

    def __init__(self, slots: int):
        self.slots = slots

    def access(self, keys: list, index: int) -> int:
        """Serve one step; return the number of misses."""
        raise NotImplementedError


class SLRU(Policy):
    """Probation and protected segments; hits move to protected, which holds
    at most half the slots. Victims come from probation first and never
    belong to the current step."""

    name = "slru"

    def __init__(
        self,
        slots: int,
        preload: frozenset = frozenset(),
        pinned: frozenset = frozenset(),
    ):
        super().__init__(slots)
        # --moe-cache-profile preloads the most used experts as protected
        # slots, which later steps may still evict.
        self.pinned = pinned
        self.probation = collections.OrderedDict()
        self.protected = collections.OrderedDict()
        for key in sorted(pinned | preload):
            self.protected[key] = None

    def _victim(self, current: set):
        for segment in (self.probation, self.protected):
            for key in segment:
                if key not in current and key not in self.pinned:
                    return segment, key
        raise RuntimeError("no victim: the cache is smaller than one step")

    def access(self, keys: list, index: int) -> int:
        current = set(keys)
        misses = 0
        for key in keys:
            if key in self.protected:
                self.protected.move_to_end(key)
            elif key in self.probation:
                del self.probation[key]
                self.protected[key] = None
            else:
                continue
            # Keep protected at half the slots, the pinned experts aside.
            while len(self.protected) - len(self.pinned) > self.slots // 2:
                for old in self.protected:
                    if old not in self.pinned and old not in current:
                        del self.protected[old]
                        self.probation[old] = None
                        break
                else:
                    break
        for key in sorted(k for k in keys if k not in self.protected and k not in self.probation):
            misses += 1
            if len(self.protected) + len(self.probation) >= self.slots:
                segment, victim = self._victim(current)
                del segment[victim]
            self.probation[key] = None
        return misses


class LRU(Policy):
    name = "lru"

    def __init__(self, slots: int):
        super().__init__(slots)
        self.order = collections.OrderedDict()

    def access(self, keys: list, index: int) -> int:
        current = set(keys)
        misses = 0
        for key in keys:
            if key in self.order:
                self.order.move_to_end(key)
                continue
            misses += 1
            if len(self.order) >= self.slots:
                for old in self.order:
                    if old not in current:
                        del self.order[old]
                        break
            self.order[key] = None
        return misses


class LFU(Policy):
    """Evicts the cached expert with the lowest decayed count, as
    --moe-cache-policy lfu does: a step first counts its experts, and every
    `decay_every` steps (layer lookups) halve all counts."""

    name = "lfu"

    def __init__(self, slots: int, decay_every: int = 4096):
        super().__init__(slots)
        self.decay_every = decay_every
        self.steps = 0
        self.score = collections.defaultdict(float)
        self.cached = set()

    def access(self, keys: list, index: int) -> int:
        for key in keys:
            self.score[key] += 1.0
        self.steps += 1
        if self.steps % self.decay_every == 0:
            for key in self.score:
                self.score[key] *= 0.5
        current = set(keys)
        misses = 0
        for key in keys:
            if key in self.cached:
                continue
            misses += 1
            if len(self.cached) >= self.slots:
                victim = min((k for k in self.cached if k not in current), key=self.score.__getitem__)
                self.cached.remove(victim)
            self.cached.add(key)
        return misses


class NextUse(Policy):
    """Evicts the expert whose next use is expected furthest away, as
    --moe-cache-policy next does. A pass runs through the layers in order, so
    an expert of layer l is next needed d lookups from now, d being the
    distance from the current layer to l (a full pass for the layer that just
    ran), plus a pass for each token it is expected to sit out. The chance a
    token uses it, p, comes from the decayed count of the LFU policy: with
    counts halved every `decay_every` lookups over n_layers layers, a count
    covers about decay_every / (n_layers ln 2) tokens."""

    name = "next"

    def __init__(self, slots: int, layers: list, decay_every: int = 4096):
        super().__init__(slots)
        self.position = {layer: i for i, layer in enumerate(sorted(layers))}
        self.n_layers = len(self.position)
        self.decay_every = decay_every
        self.window = decay_every / (self.n_layers * 0.6931471805599453)
        self.steps = 0
        self.score = collections.defaultdict(float)
        self.cached = set()

    def expected_next_use(self, key, layer_pos: int) -> float:
        distance = (self.position[key[0]] - layer_pos) % self.n_layers or self.n_layers
        p = min(1.0, self.score[key] / self.window)
        if p <= 0.0:
            return float("inf")
        return distance + self.n_layers * (1.0 / p - 1.0)

    def access(self, keys: list, index: int) -> int:
        for key in keys:
            self.score[key] += 1.0
        self.steps += 1
        if self.steps % self.decay_every == 0:
            for key in self.score:
                self.score[key] *= 0.5
        current = set(keys)
        layer_pos = self.position[keys[0][0]] if keys else 0
        misses = 0
        for key in keys:
            if key in self.cached:
                continue
            misses += 1
            if len(self.cached) >= self.slots:
                victim = max(
                    (k for k in self.cached if k not in current),
                    key=lambda k: self.expected_next_use(k, layer_pos),
                )
                self.cached.remove(victim)
            self.cached.add(key)
        return misses


class Belady(Policy):
    """Offline optimum: needs the whole trace up front."""

    name = "belady"

    def __init__(self, slots: int, trace_keys: list):
        super().__init__(slots)
        self.next_use = []
        last = {}
        for i in range(len(trace_keys) - 1, -1, -1):
            uses = {}
            for key in trace_keys[i]:
                uses[key] = last.get(key, float("inf"))
            self.next_use.append(uses)
            for key in trace_keys[i]:
                last[key] = i
        self.next_use.reverse()
        self.cached = {}  # key -> next use
        self.heap = []  # (-next use, key), lazily invalidated

    def access(self, keys: list, index: int) -> int:
        current = set(keys)
        misses = 0
        for key in keys:
            nxt = self.next_use[index][key]
            if key not in self.cached:
                misses += 1
                if len(self.cached) >= self.slots:
                    deferred = []
                    while True:
                        neg, victim = heapq.heappop(self.heap)
                        if self.cached.get(victim) != -neg:
                            continue
                        if victim in current:
                            deferred.append((neg, victim))
                            continue
                        del self.cached[victim]
                        break
                    for item in deferred:
                        heapq.heappush(self.heap, item)
            self.cached[key] = nxt
            heapq.heappush(self.heap, (-nxt, key))
        return misses


def profile_pins(path: str, slots: int, share: float) -> frozenset:
    """The experts a profile trace selected most, up to share of the slots."""
    counts = collections.Counter()
    for step in read_trace(path):
        for e in step.experts:
            counts[(step.layer, e)] += 1
    return frozenset(key for key, _ in counts.most_common(int(slots * share)))


@dataclass
class Result:
    name: str
    misses: int = 0
    selections: int = 0
    tokens: int = 0


def simulate(steps: list, policies: list) -> list:
    trace_keys = [unique(step) for step in steps]
    results = []
    for policy in policies:
        result = Result(policy.name)
        for index, (step, keys) in enumerate(zip(steps, trace_keys)):
            result.misses += policy.access(keys, index)
            result.selections += len(keys)
        result.tokens = sum(s.tokens for s in steps) // max(1, len({s.layer for s in steps}))
        results.append(result)
    return results


def skew(steps: list) -> list:
    """Share of all selections the most used experts cover."""
    counts = collections.Counter()
    for step in steps:
        for key in unique(step):
            counts[key] += 1
    total = sum(counts.values())
    ordered = [c for _, c in counts.most_common()]
    lines = []
    for share in (0.05, 0.10, 0.25, 0.50):
        n = max(1, int(len(ordered) * share))
        lines.append((share, sum(ordered[:n]) / total if total else 0.0))
    return lines


def write_synthetic(path: str, layers: int = 40, experts: int = 256, used: int = 8, tokens: int = 400, seed: int = 1) -> None:
    """A trace whose experts follow a Zipf-like popularity, independently per
    token: no temporal locality, so it compares policies on frequency alone."""
    import random

    rng = random.Random(seed)
    weights = [1.0 / (i + 1) ** 0.9 for i in range(experts)]
    with open(path, "w", encoding="utf8") as f:
        for _ in range(tokens):
            for layer in range(layers):
                chosen = set()
                while len(chosen) < used:
                    chosen.add(rng.choices(range(experts), weights)[0])
                f.write(f"{layer} 1 " + " ".join(map(str, sorted(chosen))) + "\n")


def self_test() -> None:
    # Two layers; a hot set of 4 experts and a cold tail.
    import random

    rng = random.Random(0)
    steps = []
    for _ in range(2000):
        for layer in (0, 1):
            hot = rng.sample(range(4), 2)
            cold = rng.sample(range(4, 64), 2)
            steps.append(Step(layer, 1, hot + cold))
    trace_keys = [unique(s) for s in steps]
    results = {r.name: r for r in simulate(steps, [SLRU(16), LRU(16), LFU(16), NextUse(16, [0, 1]), Belady(16, trace_keys)])}
    best = results["belady"].misses
    for name, r in results.items():
        assert r.misses >= best, (name, r.misses, best)
    assert results["lfu"].misses < results["lru"].misses, "a stable hot set favours frequency"
    print("self-test passed")


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", nargs="?")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--slots", type=int, help="experts the cache holds")
    size.add_argument("--fraction", type=float, help="share of the traced layers' experts the cache holds")
    parser.add_argument("--experts", type=int, help="experts per layer (default: the largest id + 1)")
    parser.add_argument("--profile", help="an earlier trace whose most used experts the warm policy preloads")
    parser.add_argument("--pin-share", type=float, default=0.5, help="share of the slots the warm policy preloads (default 0.5, as the engine)")
    parser.add_argument("--expert-bytes", type=int, help="bytes of one expert (all tensors), to report the uploads")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="write a synthetic trace (40 layers, 256 experts, 8 per token, 400 tokens) to TRACE first",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    if args.synthetic:
        if not args.trace:
            parser.error("--synthetic needs a TRACE path to write")
        write_synthetic(args.trace)
    if not args.trace:
        parser.error("a trace is required")

    steps = read_trace(args.trace)
    if not steps:
        print("the trace is empty", file=sys.stderr)
        return 1
    layers = sorted({s.layer for s in steps})
    n_expert = args.experts or 1 + max(e for s in steps for e in s.experts)
    slots = args.slots or int((args.fraction or 0.25) * n_expert * len(layers))
    largest_step = max(len(set(s.experts)) for s in steps)
    if slots < largest_step:
        print(f"{slots} slots cannot hold the {largest_step} experts of one step", file=sys.stderr)
        return 1

    trace_keys = [unique(step) for step in steps]
    policies = [SLRU(slots), LRU(slots), LFU(slots), NextUse(slots, layers)]
    if args.profile:
        policies.append(SLRU(slots, preload=profile_pins(args.profile, slots, args.pin_share)))
        policies[-1].name = "warm"
    policies.append(Belady(slots, trace_keys))

    print(f"{len(steps)} steps over {len(layers)} layers, {n_expert} experts per layer, {slots} slots "
          f"({100.0 * slots / (n_expert * len(layers)):.1f}% of the traced experts)")
    for share, covered in skew(steps):
        print(f"  the {100 * share:.0f}% most used experts take {100 * covered:.1f}% of the selections")
    print(f"{'policy':8} {'hit rate':>9} {'misses':>10} {'misses/token':>13}" + ("  uploaded" if args.expert_bytes else ""))
    for r in simulate(steps, policies):
        hit = 1.0 - r.misses / r.selections if r.selections else 0.0
        per_token = r.misses / r.tokens if r.tokens else 0.0
        line = f"{r.name:8} {100 * hit:8.2f}% {r.misses:10d} {per_token:13.2f}"
        if args.expert_bytes:
            line += f"  {r.misses * args.expert_bytes / 2**30:8.2f} GiB"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
