from __future__ import annotations

from typing import Dict, Iterable, List, Set


def class_partitions(session_plan: List[List[int]], step: int) -> Dict[str, Set[int]]:
    step = int(step)
    old: Set[int] = set()
    for i in range(max(step, 0)):
        if i < len(session_plan):
            old.update(int(x) for x in session_plan[i])
    new = set(int(x) for x in session_plan[step]) if 0 <= step < len(session_plan) else set()
    seen = old | new
    all_classes = set()
    for group in session_plan:
        all_classes.update(int(x) for x in group)
    future = all_classes - seen
    return {
        "old": old,
        "new": new,
        "future": future,
        "seen": seen,
        "all": all_classes,
        "old_all": old,
        "new_current": new,
    }


def group_for_class(class_id: int, partitions: Dict[str, Set[int]]) -> str:
    cid = int(class_id)
    if cid in partitions.get("old", set()):
        return "old"
    if cid in partitions.get("new", set()):
        return "new"
    if cid in partitions.get("future", set()):
        return "future"
    return "all"


def parse_groups(raw: str) -> List[str]:
    groups = [x.strip() for x in str(raw or "old,new,future").split(",") if x.strip()]
    return groups or ["old", "new", "future"]

