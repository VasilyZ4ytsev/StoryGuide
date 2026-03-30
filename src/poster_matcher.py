import pickle
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.dataset_loader import load_movie_metadata


BASE_DIR = Path(__file__).resolve().parent.parent
FULL_POSTER_DIR = BASE_DIR / "data" / "raw" / "FullMoviePosters"
POSTER_INDEX_PATH = BASE_DIR / "data" / "processed" / "poster_index.pkl"
POSTER_INDEX_VERSION = 2


def _require_poster_dir():
    if not FULL_POSTER_DIR.is_dir():
        raise FileNotFoundError(
            f"Poster directory is missing: {FULL_POSTER_DIR.resolve()}"
        )

    jpg_files = sorted(FULL_POSTER_DIR.glob("*.jpg"))
    if not jpg_files:
        raise FileNotFoundError(
            f"Poster directory does not contain .jpg files: {FULL_POSTER_DIR.resolve()}"
        )
    return FULL_POSTER_DIR


def _resize_for_matching(image_bgr, max_side=384):
    height, width = image_bgr.shape[:2]
    scale = max(height, width) / float(max_side)
    if scale <= 1:
        return image_bgr.copy()

    resized_width = max(1, int(round(width / scale)))
    resized_height = max(1, int(round(height / scale)))
    return cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _center_crop(image_bgr, crop_ratio=0.82):
    height, width = image_bgr.shape[:2]
    crop_height = max(1, int(round(height * crop_ratio)))
    crop_width = max(1, int(round(width * crop_ratio)))
    top = max(0, (height - crop_height) // 2)
    left = max(0, (width - crop_width) // 2)
    return image_bgr[top:top + crop_height, left:left + crop_width]


def _prepare_gray(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _compute_dhash(image_bgr, hash_size=8):
    gray = _prepare_gray(image_bgr)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    difference = resized[:, 1:] > resized[:, :-1]
    bit_string = "".join("1" if value else "0" for value in difference.flatten())
    return int(bit_string, 2)


def _compute_phash(image_bgr, hash_size=8, highfreq_factor=4):
    gray = _prepare_gray(image_bgr)
    size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype("float32")
    dct = cv2.dct(resized)
    low_freq = dct[:hash_size, :hash_size]
    median_value = float(np.median(low_freq[1:, 1:]))
    bits = low_freq > median_value
    bit_string = "".join("1" if value else "0" for value in bits.flatten())
    return int(bit_string, 2)


def _hamming_similarity(left_hash, right_hash, bit_length=64):
    distance = (left_hash ^ right_hash).bit_count()
    return max(0.0, 1.0 - (distance / float(bit_length)))


def _compute_histogram(image_bgr):
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv_image], [0, 1], None, [16, 16], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten().astype("float32")
    return histogram


def _compute_edge_signature(image_bgr):
    gray = _prepare_gray(image_bgr)
    edges = cv2.Canny(gray, 80, 180)
    pooled = cv2.resize(edges, (24, 36), interpolation=cv2.INTER_AREA).astype("float32")
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled /= norm
    return pooled.flatten()


def _histogram_similarity(left_histogram, right_histogram):
    score = cv2.compareHist(left_histogram, right_histogram, cv2.HISTCMP_CORREL)
    return float(max(0.0, min((score + 1.0) / 2.0, 1.0)))


def _cosine_similarity(left_vector, right_vector):
    numerator = float(np.dot(left_vector, right_vector))
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    score = numerator / (left_norm * right_norm)
    return max(0.0, min(score, 1.0))


@lru_cache(maxsize=1)
def _orb():
    return cv2.ORB_create(nfeatures=700, fastThreshold=12)


@lru_cache(maxsize=1)
def _akaze():
    return cv2.AKAZE_create()


def _compute_orb_descriptors(image_bgr):
    gray = _prepare_gray(image_bgr)
    keypoints, descriptors = _orb().detectAndCompute(gray, None)
    return descriptors, len(keypoints or [])


def _compute_akaze_descriptors(image_bgr):
    gray = _prepare_gray(image_bgr)
    keypoints, descriptors = _akaze().detectAndCompute(gray, None)
    return descriptors, len(keypoints or [])


@lru_cache(maxsize=1)
def _movie_records_by_imdb_id():
    return {
        str(record.get("imdb_id", "")).strip(): record
        for record in load_movie_metadata()
        if str(record.get("imdb_id", "")).strip()
    }


def _poster_file_signature(poster_dir):
    jpg_files = sorted(poster_dir.glob("*.jpg"))
    latest_mtime = max(file_path.stat().st_mtime for file_path in jpg_files)
    return {
        "dir": str(poster_dir),
        "count": len(jpg_files),
        "latest_mtime": latest_mtime,
        "version": POSTER_INDEX_VERSION,
    }


def poster_index_stats():
    records_by_id = _movie_records_by_imdb_id()
    poster_index = load_poster_index()
    indexed_count = len(poster_index)
    catalog_count = len(records_by_id)
    coverage_ratio = (indexed_count / float(catalog_count)) if catalog_count else 0.0
    return {
        "indexed_count": indexed_count,
        "catalog_count": catalog_count,
        "coverage_ratio": round(coverage_ratio, 4),
    }


def _load_cached_index(signature):
    if not POSTER_INDEX_PATH.exists():
        return None
    try:
        with POSTER_INDEX_PATH.open("rb") as file:
            payload = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError):
        return None

    if payload.get("signature") != signature:
        return None
    return payload.get("items", [])


def _save_cached_index(signature, items):
    POSTER_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSTER_INDEX_PATH.open("wb") as file:
        pickle.dump({"signature": signature, "items": items}, file, protocol=pickle.HIGHEST_PROTOCOL)


@lru_cache(maxsize=1)
def load_poster_index():
    poster_dir = _require_poster_dir()
    signature = _poster_file_signature(poster_dir)
    cached_items = _load_cached_index(signature)
    if cached_items is not None:
        if not cached_items:
            raise RuntimeError(
                f"Poster index cache is empty for {poster_dir.resolve()}."
            )
        return cached_items

    records_by_id = _movie_records_by_imdb_id()
    index = []
    for file_path in sorted(poster_dir.glob("*.jpg")):
        imdb_id = file_path.stem
        movie_record = records_by_id.get(imdb_id)
        if movie_record is None:
            continue

        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        prepared = _resize_for_matching(image)
        center_prepared = _resize_for_matching(_center_crop(prepared))
        index.append(
            {
                "imdb_id": imdb_id,
                "record": movie_record,
                "path": str(file_path),
                "dhash": _compute_dhash(prepared),
                "phash": _compute_phash(prepared),
                "center_phash": _compute_phash(center_prepared),
                "histogram": _compute_histogram(prepared),
                "center_histogram": _compute_histogram(center_prepared),
                "edge_signature": _compute_edge_signature(prepared),
            }
        )

    if not index:
        raise RuntimeError(
            f"Poster index could not be built from {poster_dir.resolve()}: no catalog-aligned posters were indexed."
        )

    _save_cached_index(signature, index)
    return index


def _binary_descriptor_similarity(query_descriptors, candidate_descriptors):
    if query_descriptors is None or candidate_descriptors is None:
        return 0.0
    if len(query_descriptors) < 8 or len(candidate_descriptors) < 8:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good_matches.append(first)

    normalizer = max(min(len(query_descriptors), len(candidate_descriptors)), 1)
    return min(1.0, len(good_matches) / float(normalizer))


def match_movie_poster(image_bgr, top_k=3):
    poster_index = load_poster_index()

    prepared_query = _resize_for_matching(image_bgr)
    center_query = _resize_for_matching(_center_crop(prepared_query))
    query_hash = _compute_dhash(prepared_query)
    query_phash = _compute_phash(prepared_query)
    query_center_phash = _compute_phash(center_query)
    query_histogram = _compute_histogram(prepared_query)
    query_center_histogram = _compute_histogram(center_query)
    query_edge_signature = _compute_edge_signature(prepared_query)
    query_orb_descriptors, query_orb_keypoints = _compute_orb_descriptors(prepared_query)
    query_akaze_descriptors, query_akaze_keypoints = _compute_akaze_descriptors(prepared_query)
    if max(query_orb_keypoints, query_akaze_keypoints) < 8:
        return None

    prescored = []
    for candidate in poster_index:
        hash_score = _hamming_similarity(query_hash, candidate["dhash"])
        phash_score = _hamming_similarity(query_phash, candidate["phash"])
        center_phash_score = _hamming_similarity(query_center_phash, candidate["center_phash"])
        histogram_score = _histogram_similarity(query_histogram, candidate["histogram"])
        center_histogram_score = _histogram_similarity(query_center_histogram, candidate["center_histogram"])
        edge_score = _cosine_similarity(query_edge_signature, candidate["edge_signature"])
        pre_score = (
            hash_score * 0.14
            + phash_score * 0.22
            + center_phash_score * 0.22
            + histogram_score * 0.12
            + center_histogram_score * 0.14
            + edge_score * 0.16
        )
        prescored.append(
            (
                pre_score,
                hash_score,
                phash_score,
                center_phash_score,
                histogram_score,
                center_histogram_score,
                edge_score,
                candidate,
            )
        )

    prescored.sort(key=lambda item: item[0], reverse=True)
    shortlist = prescored[:140]

    matches = []
    for (
        pre_score,
        hash_score,
        phash_score,
        center_phash_score,
        histogram_score,
        center_histogram_score,
        edge_score,
        candidate,
    ) in shortlist:
        candidate_image = cv2.imread(candidate["path"], cv2.IMREAD_COLOR)
        if candidate_image is None:
            continue
        candidate_prepared = _resize_for_matching(candidate_image)
        candidate_orb_descriptors, _ = _compute_orb_descriptors(candidate_prepared)
        candidate_akaze_descriptors, _ = _compute_akaze_descriptors(candidate_prepared)
        orb_score = _binary_descriptor_similarity(query_orb_descriptors, candidate_orb_descriptors)
        akaze_score = _binary_descriptor_similarity(query_akaze_descriptors, candidate_akaze_descriptors)
        local_score = max(orb_score, akaze_score) * 0.55 + min(orb_score, akaze_score) * 0.15
        final_score = pre_score * 0.42 + local_score * 0.58
        matches.append(
            {
                "record": candidate["record"],
                "score": round(final_score, 4),
                "pre_score": round(pre_score, 4),
                "orb_score": round(orb_score, 4),
                "akaze_score": round(akaze_score, 4),
                "hash_score": round(hash_score, 4),
                "phash_score": round(phash_score, 4),
                "center_phash_score": round(center_phash_score, 4),
                "histogram_score": round(histogram_score, 4),
                "center_histogram_score": round(center_histogram_score, 4),
                "edge_score": round(edge_score, 4),
            }
        )

    matches.sort(key=lambda item: item["score"], reverse=True)
    best_match = matches[0] if matches else None
    if best_match is None or best_match["score"] < 0.35:
        return None

    return {
        "best_match": best_match,
        "matches": matches[:top_k],
    }
