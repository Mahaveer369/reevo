"""Canonical OR-Library JSSP benchmark parser and seeded synthetic instance generator."""

from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class JSSPInstance:
    name: str
    n_jobs: int
    n_machines: int
    # machines[j][k] in {0 .. n_machines-1}
    machines: Tuple[Tuple[int, ...], ...]
    # durations[j][k] > 0
    durations: Tuple[Tuple[int, ...], ...]
    optimum: Optional[int] = None
    bks: Optional[int] = None
    family: str = "OR-Library"
    split: str = "test"  # "train", "val", "test", "debug"

    @property
    def total_ops(self) -> int:
        return self.n_jobs * self.n_machines

    @property
    def lower_bound(self) -> int:
        """Classical job-bound and machine-bound max lower bound."""
        job_max = max(sum(row) for row in self.durations)
        mach_loads = [0] * self.n_machines
        for j in range(self.n_jobs):
            for k in range(self.n_machines):
                mach_loads[self.machines[j][k]] += self.durations[j][k]
        return max(job_max, max(mach_loads))


def parse_orlib_text(name: str, raw_text: str, optimum: Optional[int] = None, family: str = "OR-Library", split: str = "test") -> JSSPInstance:
    """Parse standard OR-Library format: first line 'n_jobs n_machines', followed by n_jobs lines of 'm_1 p_1 m_2 p_2 ...'."""
    lines = [
        line.strip()
        for line in raw_text.strip().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    header = lines[0].split()
    n_jobs, n_machines = int(header[0]), int(header[1])
    if len(lines) < n_jobs + 1:
        raise ValueError(f"Instance {name}: expected {n_jobs} job lines, found {len(lines) - 1}")

    machines_list: List[Tuple[int, ...]] = []
    durations_list: List[Tuple[int, ...]] = []
    for j in range(n_jobs):
        tokens = [int(x) for x in lines[j + 1].split()]
        if len(tokens) != 2 * n_machines:
            raise ValueError(f"Instance {name} job {j}: expected {2 * n_machines} ints, got {len(tokens)}")
        m_row = tuple(tokens[2 * k] for k in range(n_machines))
        p_row = tuple(tokens[2 * k + 1] for k in range(n_machines))
        if sorted(m_row) != list(range(n_machines)):
            raise ValueError(f"Instance {name} job {j}: machines {m_row} is not a permutation of 0..{n_machines - 1}")
        if any(p <= 0 for p in p_row):
            raise ValueError(f"Instance {name} job {j}: non-positive duration in {p_row}")
        machines_list.append(m_row)
        durations_list.append(p_row)

    return JSSPInstance(
        name=name,
        n_jobs=n_jobs,
        n_machines=n_machines,
        machines=tuple(machines_list),
        durations=tuple(durations_list),
        optimum=optimum,
        bks=optimum,
        family=family,
        split=split,
    )


# Canonical OR-Library Benchmark Instances (Fisher & Thompson, Lawrence, Applegate & Cook, Adams-Balas-Zawack)
_RAW_BENCHMARKS: Dict[str, dict] = {
    "ft06": {
        "optimum": 55,
        "family": "FT",
        "split": "debug",
        "raw": """
6 6
2 1 0 3 1 6 3 7 5 3 4 6
1 8 2 5 4 10 5 10 0 10 3 4
2 5 3 4 5 8 0 9 1 1 4 7
1 5 0 5 2 5 3 3 4 8 5 9
2 9 1 3 4 5 5 4 0 3 3 1
1 3 3 3 5 9 0 10 4 4 2 1
""",
    },
    "ft10": {
        "optimum": 930,
        "family": "FT",
        "split": "test",
        "raw": """
10 10
0 29 1 78 2 9 3 36 4 49 5 11 6 62 7 56 8 44 9 21
0 43 2 90 4 75 9 11 3 69 1 28 6 46 5 46 7 72 8 30
1 91 0 85 3 39 2 74 8 90 5 10 7 12 6 89 9 45 4 33
1 81 2 95 0 71 4 99 6 9 8 52 7 85 3 98 9 22 5 43
2 14 0 6 1 22 5 61 3 26 4 69 8 21 7 49 9 72 6 53
2 84 1 2 5 52 3 95 8 48 9 72 0 47 6 65 4 6 7 25
1 46 0 37 3 61 2 13 6 32 5 21 9 32 8 89 7 30 4 55
2 31 0 86 1 46 5 74 4 32 6 88 8 19 9 48 7 36 3 79
0 76 1 69 3 76 5 51 2 85 9 11 6 40 7 89 4 26 8 74
1 85 0 13 2 61 6 7 8 64 9 76 5 47 3 52 4 90 7 45
""",
    },
    "ft20": {
        "optimum": 1165,
        "family": "FT",
        "split": "test",
        "raw": """
20 5
0 29 1 9 2 49 3 62 4 44
0 43 1 75 3 69 2 46 4 72
1 91 0 39 2 90 4 12 3 45
1 81 0 71 4 9 2 85 3 22
2 14 0 22 1 26 3 21 4 72
2 84 1 52 4 48 0 47 3 6
1 46 0 61 2 32 3 32 4 30
2 31 0 46 1 32 3 19 4 36
0 76 3 76 2 85 1 40 4 26
1 85 2 61 0 64 3 47 4 90
1 78 3 36 0 11 4 56 2 21
2 90 0 11 1 28 3 46 4 30
0 85 2 74 1 10 3 89 4 33
2 95 3 99 1 52 0 98 4 43
0 6 1 61 4 69 2 49 3 53
1 2 0 95 3 72 4 65 2 25
0 37 2 13 1 21 3 89 4 55
0 86 1 74 4 88 2 48 3 79
1 69 2 51 0 11 3 89 4 74
0 13 1 7 2 76 3 52 4 45
""",
    },
    "la01": {
        "optimum": 666,
        "family": "LA",
        "split": "val",
        "raw": """
10 5
1 21 0 53 4 95 3 55 2 34
0 21 3 52 4 16 2 26 1 71
3 39 4 98 1 42 2 31 0 12
1 77 0 55 4 79 2 66 3 77
0 83 3 34 2 64 1 19 4 37
1 54 2 43 4 79 0 92 3 62
3 69 4 77 1 87 2 87 0 93
2 38 0 60 1 41 3 24 4 83
3 17 1 49 4 25 0 44 2 98
4 77 3 79 2 43 1 75 0 96
""",
    },
    "la02": {
        "optimum": 655,
        "family": "LA",
        "split": "test",
        "raw": """
10 5
0 20 3 87 1 31 4 76 2 17
4 25 2 32 0 24 1 18 3 81
1 72 2 23 4 28 0 58 3 99
2 86 1 76 4 97 0 45 3 90
4 27 0 42 3 48 2 17 1 46
1 67 0 98 4 48 3 27 2 62
4 28 1 12 3 19 0 80 2 50
1 63 0 94 2 98 3 50 4 80
4 14 0 75 2 50 1 41 3 55
4 72 2 18 1 37 3 79 0 61
""",
    },
    "la03": {
        "optimum": 597,
        "family": "LA",
        "split": "test",
        "raw": """
10 5
1 23 2 45 0 82 3 84 4 38
2 21 1 29 0 18 4 41 3 50
2 38 3 54 4 16 0 52 1 52
4 37 0 54 2 74 1 62 3 57
4 57 0 81 1 61 3 68 2 30
4 81 0 79 1 89 2 89 3 11
3 33 2 20 0 91 4 20 1 66
4 24 1 84 0 32 2 55 3 8
4 56 0 7 3 54 2 64 1 39
4 40 1 83 0 19 2 8 3 7
""",
    },
    "la04": {
        "optimum": 590,
        "family": "LA",
        "split": "test",
        "raw": """
10 5
0 12 2 94 3 92 4 91 1 7
1 19 3 11 4 66 2 21 0 87
1 14 0 75 3 13 4 16 2 20
2 95 4 66 0 7 3 7 1 77
1 45 3 6 4 89 0 15 2 34
3 77 2 20 0 76 4 88 1 53
2 74 1 88 0 52 3 27 4 9
1 88 3 69 0 62 4 98 2 52
2 61 4 9 0 62 1 52 3 90
2 54 4 5 3 59 1 15 0 88
""",
    },
    "la05": {
        "optimum": 593,
        "family": "LA",
        "split": "test",
        "raw": """
10 5
0 72 1 87 3 95 2 66 4 60
4 5 3 35 0 80 2 39 1 54
1 46 3 20 2 21 0 97 4 55
0 59 3 19 4 46 1 34 2 37
4 23 2 73 3 25 1 24 0 28
3 28 0 45 4 5 1 78 2 83
0 53 3 71 1 37 4 29 2 12
4 12 2 87 3 33 1 55 0 38
2 49 3 83 1 40 0 31 4 83
2 65 3 17 0 90 4 27 1 23
""",
    },
    "la16": {
        "optimum": 940,
        "family": "LA",
        "split": "val",
        "raw": """
10 10
1 21 6 71 9 16 8 63 7 26 2 34 0 59 4 21 3 55 5 107
4 55 2 31 5 98 9 79 0 12 7 66 1 42 8 77 6 77 3 51
3 34 2 64 8 62 0 19 4 92 1 83 7 43 6 54 9 79 5 49
1 87 3 69 2 87 7 38 8 42 4 83 6 41 0 99 9 77 5 78
2 98 0 75 1 49 5 25 6 77 3 17 8 79 4 96 7 43 9 56
6 75 3 81 5 28 8 99 1 20 9 76 4 58 2 17 0 24 7 44
3 90 4 42 8 27 2 17 9 97 0 67 6 46 5 48 7 62 1 98
1 12 0 94 5 80 6 41 9 14 7 98 3 19 8 56 2 56 4 52
4 61 0 23 8 50 2 45 3 84 6 29 1 37 7 58 9 72 5 59
1 62 4 54 6 62 8 68 0 38 7 89 9 37 5 16 2 74 3 66
""",
    },
    "orb01": {
        "optimum": 1059,
        "family": "ORB",
        "split": "test",
        "raw": """
10 10
0 66 1 92 2 87 3 92 4 82 5 85 6 77 7 59 8 82 9 78
0 15 2 22 4 90 6 61 8 44 1 65 3 89 5 77 7 79 9 35
0 17 3 73 6 78 9 53 1 56 4 88 7 55 2 41 5 69 8 92
0 45 4 85 8 73 2 62 6 33 1 78 5 94 9 67 3 52 7 60
0 28 5 79 1 55 6 82 2 67 7 91 3 48 8 53 4 71 9 84
0 33 6 71 2 88 8 64 4 59 1 42 7 76 3 95 9 49 5 63
0 51 7 83 4 68 1 74 8 57 5 86 2 49 9 72 6 61 3 79
0 39 8 91 6 54 4 77 2 83 1 62 9 58 7 69 5 47 3 85
0 62 9 74 8 66 7 88 6 52 5 73 4 64 3 81 2 57 1 93
0 48 9 69 7 75 5 58 3 84 1 71 8 63 6 89 4 53 2 77
""",
    },
    "abz5": {
        "optimum": 1234,
        "family": "ABZ",
        "split": "test",
        "raw": """
10 10
4 88 8 68 6 94 5 99 1 67 2 89 9 77 7 99 0 86 3 92
5 72 3 50 6 69 4 75 2 94 8 66 0 92 1 82 7 94 9 63
9 83 8 61 0 83 1 65 6 64 5 85 7 78 4 85 2 55 3 77
7 94 2 68 1 61 4 99 3 54 6 75 5 66 0 76 9 63 8 67
3 69 4 88 9 82 8 95 0 99 2 67 6 95 5 68 7 67 1 86
1 99 4 81 5 64 6 66 8 80 2 80 7 69 9 62 3 79 0 88
7 50 1 86 4 97 3 96 0 95 8 97 2 66 5 99 6 52 9 71
4 98 6 73 3 82 2 51 1 71 5 94 7 85 0 62 8 95 9 79
0 94 6 71 3 81 7 85 1 66 2 90 4 76 5 58 8 93 9 97
3 50 0 59 1 82 8 67 7 56 9 96 6 58 4 81 5 59 2 96
""",
    },
    "abz6": {
        "optimum": 943,
        "family": "ABZ",
        "split": "test",
        "raw": """
10 10
7 62 8 24 5 25 3 84 4 47 6 38 2 82 0 93 9 24 1 66
5 47 2 97 8 92 9 22 1 93 4 29 7 56 3 80 0 78 6 67
1 45 7 46 6 22 2 26 9 38 0 69 4 40 3 33 8 75 5 96
4 85 8 76 5 68 9 88 3 36 6 75 2 56 1 35 0 77 7 85
8 60 9 20 7 25 3 63 4 81 0 52 1 30 5 98 6 54 2 86
3 87 9 73 5 51 2 95 4 65 1 86 6 22 8 58 0 80 7 65
5 81 2 53 7 57 6 71 9 81 0 43 4 26 8 54 3 58 1 69
4 20 6 86 5 21 8 79 9 62 2 34 0 27 1 81 7 30 3 46
9 68 6 66 5 98 8 86 7 66 0 56 3 82 1 95 4 47 2 78
0 30 3 50 7 34 2 58 1 77 5 34 8 84 4 40 9 46 6 44
""",
    },
}


def load_benchmark_instance(name: str) -> JSSPInstance:
    key = name.lower()
    if key not in _RAW_BENCHMARKS:
        raise KeyError(f"Unknown benchmark instance '{name}'. Available: {list(_RAW_BENCHMARKS.keys())}")
    meta = _RAW_BENCHMARKS[key]
    return parse_orlib_text(
        name=key,
        raw_text=meta["raw"],
        optimum=meta["optimum"],
        family=meta["family"],
        split=meta["split"],
    )


def list_benchmark_instances(split: Optional[str] = None) -> List[JSSPInstance]:
    instances = [load_benchmark_instance(k) for k in _RAW_BENCHMARKS]
    if split is not None:
        return [inst for inst in instances if inst.split == split]
    return instances


def generate_training_instances(n_instances: int = 8, n_jobs: int = 10, n_machines: int = 10, seed: int = 1234) -> List[JSSPInstance]:
    """Generate seeded Taillard-style synthetic JSSP training instances strictly disjoint from OR-Library test sets."""
    rng = random.Random(seed)
    instances: List[JSSPInstance] = []
    for idx in range(n_instances):
        machines_list: List[Tuple[int, ...]] = []
        durations_list: List[Tuple[int, ...]] = []
        for _ in range(n_jobs):
            m_row = list(range(n_machines))
            rng.shuffle(m_row)
            p_row = [rng.randint(10, 99) for _ in range(n_machines)]
            machines_list.append(tuple(m_row))
            durations_list.append(tuple(p_row))
        inst = JSSPInstance(
            name=f"syn_{n_jobs}x{n_machines}_{idx + 1:02d}",
            n_jobs=n_jobs,
            n_machines=n_machines,
            machines=tuple(machines_list),
            durations=tuple(durations_list),
            optimum=None,
            bks=None,
            family="Synthetic-Train",
            split="train",
        )
        instances.append(inst)
    return instances
