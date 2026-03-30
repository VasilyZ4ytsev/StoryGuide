import argparse
import concurrent.futures
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.dataset_loader import load_movie_metadata


BASE_DIR = Path(__file__).resolve().parent.parent
FULL_POSTER_DIR = BASE_DIR / "data" / "raw" / "FullMoviePosters"
FAILURES_PATH = BASE_DIR / "data" / "processed" / "poster_download_failures.txt"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StoryGuide/1.0"


def _target_path(record):
    imdb_id = str(record.get("imdb_id", "")).strip()
    if not imdb_id:
        return None
    return FULL_POSTER_DIR / f"{imdb_id}.jpg"


def _download_one(task):
    record, timeout = task
    target_path = _target_path(record)
    if target_path is None:
        return "skipped", "missing_imdb_id"

    poster_url = str(record.get("poster_url", "")).strip()
    if not poster_url:
        return "skipped", target_path.name

    if target_path.exists() and target_path.stat().st_size > 0:
        return "exists", target_path.name

    request = urllib.request.Request(
        poster_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        },
    )

    temp_path = target_path.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
        if not content:
            return "failed", target_path.name
        temp_path.write_bytes(content)
        temp_path.replace(target_path)
        return "downloaded", target_path.name
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return "failed", target_path.name


def download_posters(limit=None, workers=12, timeout=20):
    FULL_POSTER_DIR.mkdir(parents=True, exist_ok=True)
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = [record for record in load_movie_metadata() if record.get("poster_url") and record.get("imdb_id")]
    if limit is not None:
        records = records[:limit]

    counters = {"downloaded": 0, "exists": 0, "failed": 0, "skipped": 0}
    failures = []
    processed = 0
    total = len(records)
    lock = threading.Lock()
    started_at = time.time()

    def consume(result):
        nonlocal processed
        status, detail = result
        with lock:
            counters[status] = counters.get(status, 0) + 1
            processed += 1
            if status == "failed":
                failures.append(detail)
            if processed % 200 == 0 or processed == total:
                elapsed = max(time.time() - started_at, 0.001)
                speed = processed / elapsed
                print(
                    f"processed={processed}/{total} "
                    f"downloaded={counters['downloaded']} "
                    f"exists={counters['exists']} "
                    f"failed={counters['failed']} "
                    f"speed={speed:.2f}/s"
                )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download_one, (record, timeout)) for record in records]
        for future in concurrent.futures.as_completed(futures):
            consume(future.result())

    if failures:
        FAILURES_PATH.write_text("\n".join(sorted(set(failures))), encoding="utf-8")
    elif FAILURES_PATH.exists():
        FAILURES_PATH.unlink()

    return {
        "total": total,
        **counters,
        "failures_path": str(FAILURES_PATH) if failures else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Download poster images for MovieGenre.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    result = download_posters(limit=args.limit, workers=args.workers, timeout=args.timeout)
    print(result)


if __name__ == "__main__":
    main()
