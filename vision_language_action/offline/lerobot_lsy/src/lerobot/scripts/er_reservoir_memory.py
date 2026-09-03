#!/usr/bin/env python

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def load_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    with open(path) as handle:
        manifest = json.load(handle)
    validate_manifest(manifest, path)
    return manifest


def validate_manifest(manifest: dict[str, Any], path: Path | None = None) -> None:
    source = f" in {path}" if path else ""
    if manifest.get("unit") != "episode":
        raise ValueError(f"Unsupported reservoir unit{source}: {manifest.get('unit')}")
    if manifest.get("strategy") != "reservoir":
        raise ValueError(f"Unsupported reservoir strategy{source}: {manifest.get('strategy')}")
    if not isinstance(manifest.get("capacity"), int) or manifest["capacity"] <= 0:
        raise ValueError(f"Invalid reservoir capacity{source}: {manifest.get('capacity')}")
    if not isinstance(manifest.get("num_seen_episodes"), int) or manifest["num_seen_episodes"] < 0:
        raise ValueError(f"Invalid num_seen_episodes{source}: {manifest.get('num_seen_episodes')}")
    if not isinstance(manifest.get("items"), list):
        raise ValueError(f"Invalid reservoir items list{source}")
    if len(manifest["items"]) > manifest["capacity"]:
        raise ValueError(f"Reservoir items exceed capacity{source}")
    for item in manifest["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("repo_id"), str) or not isinstance(item.get("episode"), int):
            raise ValueError(f"Invalid reservoir item{source}: {item}")


def list_repo_episodes(repo_id: str, root: str | None, revision: str | None) -> list[int]:
    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    return sorted(int(ep_idx) for ep_idx in meta.episodes.keys())


def update_reservoir(
    manifest: dict[str, Any] | None,
    *,
    new_repo_id: str,
    new_episodes: list[int],
    capacity: int,
    seed: int,
) -> dict[str, Any]:
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    if manifest is None:
        items: list[dict[str, Any]] = []
        num_seen_episodes = 0
    else:
        validate_manifest(manifest)
        if manifest["capacity"] != capacity:
            raise ValueError(
                f"Cannot resume reservoir with different capacity: manifest={manifest['capacity']} requested={capacity}"
            )
        if manifest.get("seed") != seed:
            raise ValueError(f"Cannot resume reservoir with different seed: manifest={manifest.get('seed')} requested={seed}")
        items = [dict(item) for item in manifest["items"]]
        num_seen_episodes = int(manifest["num_seen_episodes"])

    for episode in new_episodes:
        num_seen_episodes += 1
        candidate = {"repo_id": new_repo_id, "episode": int(episode)}
        if len(items) < capacity:
            items.append(candidate)
        else:
            rng = random.Random(f"{seed}:{num_seen_episodes}")
            replacement_idx = rng.randrange(num_seen_episodes)
            if replacement_idx < capacity:
                items[replacement_idx] = candidate

    items = sorted(items, key=lambda item: (item["repo_id"], item["episode"]))
    counts = Counter(item["repo_id"] for item in items)
    return {
        "capacity": capacity,
        "unit": "episode",
        "strategy": "reservoir",
        "seed": seed,
        "num_seen_episodes": num_seen_episodes,
        "items": items,
        "repo_counts": dict(sorted(counts.items())),
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update an episode-level reservoir replay manifest.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("--previous-manifest", type=Path, default=None)
    update_parser.add_argument("--output-manifest", type=Path, required=True)
    update_parser.add_argument("--new-repo-id", required=True)
    update_parser.add_argument("--capacity", type=int, default=50)
    update_parser.add_argument("--seed", type=int, required=True)
    update_parser.add_argument("--root", default=None)
    update_parser.add_argument("--revision", default=None)
    update_parser.add_argument("--allow-missing-previous", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command != "update":
        raise ValueError(f"Unsupported command: {args.command}")

    previous_manifest = None
    if args.previous_manifest is not None:
        previous_manifest = load_manifest(args.previous_manifest)
        if previous_manifest is None and not args.allow_missing_previous:
            raise FileNotFoundError(f"Previous reservoir manifest not found: {args.previous_manifest}")

    new_episodes = list_repo_episodes(args.new_repo_id, args.root, args.revision)
    if not new_episodes:
        raise ValueError(f"No episodes found for repo: {args.new_repo_id}")

    manifest = update_reservoir(
        previous_manifest,
        new_repo_id=args.new_repo_id,
        new_episodes=new_episodes,
        capacity=args.capacity,
        seed=args.seed,
    )
    write_manifest(manifest, args.output_manifest)
    print(
        f"Wrote reservoir manifest to {args.output_manifest} "
        f"with {len(manifest['items'])}/{manifest['capacity']} episodes, "
        f"num_seen_episodes={manifest['num_seen_episodes']}, repo_counts={manifest['repo_counts']}"
    )


if __name__ == "__main__":
    main()
