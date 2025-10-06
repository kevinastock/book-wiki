import functools
from collections import defaultdict
from dataclasses import dataclass
from sqlite3 import Cursor

import numpy as np
from rapidfuzz import fuzz, process, utils

from bookwiki.models import WikiPage


@functools.lru_cache(maxsize=100)
def _compute_similarity_scores(
    queries: tuple[str, ...], choices: tuple[str, ...]
) -> np.ndarray:
    """Compute similarity scores between queries and choices using rapidfuzz.

    This function is cached to avoid recomputing scores for repeated searches
    with the same queries and choices (e.g., when paginating through results).

    Args:
        queries: Tuple of search query strings
        choices: Tuple of available choice strings to match against

    Returns:
        2D numpy array where scores[i][j] is the similarity between
        queries[i] and choices[j]
    """
    # TODO: try WRatio when we get llm to stop making up stupid wiki pages.
    return process.cdist(
        queries,
        choices,
        scorer=fuzz.ratio,  # Don't tokenize individual names
        processor=utils.default_process,  # Lowercase and trim whitespace
    )


def _convert_name_scores_to_slug_scores(
    name_scores: np.ndarray,
    choices: list[str],
    name_to_slugs: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    """Convert similarity scores from query→name to query→slug.

    Since multiple names can map to the same slug (e.g., character aliases),
    we need to aggregate scores. For each query/slug pair, we take the MAX
    score across all names that map to that slug.

    Args:
        name_scores: 2D array where name_scores[i][j] is the similarity
            between query i and name j
        choices: List of names (in same order as name_scores columns)
        name_to_slugs: Dict mapping each name to list of slugs it belongs to

    Returns:
        Tuple of:
        - 2D array where result[i][j] is the max similarity between
          query i and any name belonging to slug j
        - List of slugs (in same order as result columns)
    """
    # Get all unique slugs in a consistent order
    slugs = sorted({slug for lst in name_to_slugs.values() for slug in lst})

    # Build a mapping from slug to the indices of names that map to it
    slug_to_name_indices: dict[str, list[int]] = defaultdict(list)
    for name_idx, name in enumerate(choices):
        for slug in name_to_slugs[name]:
            slug_to_name_indices[slug].append(name_idx)

    # Create query→slug scores matrix
    num_queries = name_scores.shape[0]
    query_slug_scores = np.zeros((num_queries, len(slugs)))

    for slug_idx, slug in enumerate(slugs):
        name_indices = slug_to_name_indices[slug]
        # For each query, find the best matching name for this slug
        for query_idx in range(num_queries):
            # Take the maximum score across all names that map to this slug
            query_slug_scores[query_idx, slug_idx] = max(
                name_scores[query_idx, name_idx] for name_idx in name_indices
            )

    return query_slug_scores, slugs


def _rank_slugs_by_query(query_slug_scores: np.ndarray) -> list[list[int]]:
    """Rank slugs for each query based on similarity scores.

    For each query, creates a ranking of slugs from best match (rank 1)
    to worst match (rank N). Higher scores get better (lower) ranks.

    Args:
        query_slug_scores: 2D array where query_slug_scores[i][j] is the
            similarity between query i and slug j

    Returns:
        List of rankings, where rankings[i] is a list of slug indices
        ranked by score for query i (best to worst)
    """
    rankings = []
    for query_idx in range(query_slug_scores.shape[0]):
        # Get scores for this query
        scores = query_slug_scores[query_idx, :]
        # Sort slug indices by score (descending = best first)
        ranked_indices = np.argsort(-scores)  # Negative for descending sort
        rankings.append(ranked_indices.tolist())

    return rankings


def _reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Combine multiple rankings using Reciprocal Rank Fusion.

    RRF formula: RRF(d) = Σ(1 / (k + rank(d, q))) for all queries q
    where rank(d, q) is the rank of document d in query q's ranking.

    Args:
        rankings: List of rankings, where rankings[i] is a list of slug indices
            ranked for query i (best to worst)
        k: RRF parameter (typically 60), controls the importance of top ranks

    Returns:
        List of (slug_index, rrf_score) tuples sorted by RRF score (best first)
    """
    if not rankings:
        return []

    # Find all unique slug indices across all rankings
    all_slug_indices = set()
    for ranking in rankings:
        all_slug_indices.update(ranking)

    # Calculate RRF score for each slug
    rrf_scores = {}
    for slug_idx in all_slug_indices:
        rrf_score = 0.0
        for ranking in rankings:
            try:
                # Find the rank of this slug in this query's ranking
                # rank is 1-based (1 = best, 2 = second best, etc.)
                rank = ranking.index(slug_idx) + 1
                rrf_score += 1.0 / (k + rank)
            except ValueError:
                # Slug not found in this ranking, contributes 0 to RRF score
                pass
        rrf_scores[slug_idx] = rrf_score

    # Sort by RRF score (descending = best first)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


@dataclass(frozen=True)
class SearchResult:
    page: WikiPage
    rank: int
    score: float


@dataclass(frozen=True)
class SearchResults:
    results: list[SearchResult]
    total_results: int
    results_page: int
    total_pages: int


def find_similar_slugs(
    cursor: Cursor, query_slug: str, latest_chapter_id: int, limit: int = 3
) -> list[WikiPage]:
    """Find wiki pages with slugs similar to the query slug using fuzzy matching.

    Args:
        cursor: Database cursor
        query_slug: The slug to find matches for
        latest_chapter_id: Latest chapter ID to limit results to
        limit: Maximum number of suggestions to return

    Returns:
        List of WikiPage objects sorted by similarity score
    """
    # Get all existing slugs at the current chapter
    unique_slugs = list(WikiPage.get_all_slugs(cursor, latest_chapter_id))

    if not unique_slugs:
        return []

    # Use rapidfuzz to find similar slugs
    matches = process.extract(
        query_slug,
        unique_slugs,
        scorer=fuzz.ratio,
        processor=utils.default_process,
        limit=limit,
    )

    # Get WikiPage objects for the matched slugs
    results = []
    for slug, _, _ in matches:
        page = WikiPage.read_page_at(cursor, slug, latest_chapter_id)
        assert page is not None
        if page:
            results.append(page)

    return results


def search_wiki_by_name(
    cursor: Cursor,
    page: int,
    names: list[str],
    latest_chapter_id: int,
    page_size: int = 10,
) -> SearchResults:
    # Get all unique (name, slug) pairs from the database
    name_slug_pairs = WikiPage.get_name_slug_pairs(cursor, latest_chapter_id)

    # Build a dict from name to list of slugs that have that name
    name_to_slugs: dict[str, list[str]] = defaultdict(list)
    for name, slug in name_slug_pairs:
        name_to_slugs[name].append(slug)

    # Extract queries and deduplicated sorted choices
    queries = names
    choices = sorted(name_to_slugs.keys())

    # Use rapidfuzz to calculate similarity scores between queries and choices
    # cdist returns a matrix where scores[i][j] is the similarity
    # between queries[i] and choices[j]
    if not choices:
        # No wiki pages exist, return empty result
        return SearchResults([], 0, 0, 0)

    # Convert to tuples for caching (lists aren't hashable)
    queries_tuple = tuple(queries)
    choices_tuple = tuple(choices)

    # Get similarity scores (cached for repeated calls with same inputs)
    name_scores = _compute_similarity_scores(queries_tuple, choices_tuple)

    # Convert from query→name scores to query→slug scores
    query_slug_scores, slugs = _convert_name_scores_to_slug_scores(
        name_scores, choices, name_to_slugs
    )

    # Rank slugs for each query based on similarity scores
    rankings = _rank_slugs_by_query(query_slug_scores)

    # Combine rankings using Reciprocal Rank Fusion
    rrf_results = _reciprocal_rank_fusion(rankings)

    # Apply pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    page_results = rrf_results[start_idx:end_idx]

    if not page_results:
        return SearchResults([], 0, 0, 0)

    results = []
    for rank, (slug_idx, rrf_score) in enumerate(page_results, start=start_idx + 1):
        slug = slugs[slug_idx]
        wp = WikiPage.read_page_at(cursor, slug, latest_chapter_id)
        assert wp is not None
        results.append(SearchResult(wp, rank, rrf_score))

    return SearchResults(
        results, len(rrf_results), page, (len(rrf_results) + page_size - 1) // page_size
    )
