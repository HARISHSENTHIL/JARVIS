"""
Brave Search API Integration for Jarvis
Provides web search fallback when documents don't have answers
Direct implementation using httpx - no external MCP package needed
"""

import logging
from typing import List, Dict, Optional
import httpx
import json
import time
from datetime import datetime, timedelta

from .config import get_config

logger = logging.getLogger(__name__)


class BraveSearchMCP:
    """
    Brave Search integration using their Web Search API v1
    Documentation: https://api.search.brave.com/app/documentation/web-search/get-started
    """

    def __init__(self):
        self.config = get_config()
        self.api_key = self.config.web_search.brave_api_key
        self.max_results = self.config.web_search.max_results
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

        # Rate limiting: Track last request time
        self._last_request_time = None
        self._min_request_interval = 0.1  # 100ms between requests

        if not self.api_key:
            logger.warning("BRAVE_API_KEY not set - web search will be disabled")

    def is_available(self) -> bool:
        """Check if Brave Search is available"""
        return self.api_key is not None and self.config.web_search.enabled

    def _rate_limit(self):
        """Simple rate limiting to avoid hitting API limits"""
        if self._last_request_time:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_request_interval:
                sleep_time = self._min_request_interval - elapsed
                time.sleep(sleep_time)
        self._last_request_time = time.time()

    def search(
        self,
        query: str,
        count: Optional[int] = None,
        safesearch: str = "moderate",
        max_retries: int = 3
    ) -> List[Dict]:
        """
        Search the web using Brave Search API with retry logic

        Args:
            query: Search query
            count: Number of results (default from config)
            safesearch: Safe search level (off, moderate, strict)
            max_retries: Maximum number of retry attempts on failure

        Returns:
            List of search results with title, url, snippet
        """
        if not self.is_available():
            logger.warning("Brave Search not available - check BRAVE_API_KEY in .env")
            return []

        count = count or self.max_results

        # Apply rate limiting
        self._rate_limit()

        for attempt in range(max_retries):
            try:
                headers = {
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.api_key
                }

                params = {
                    "q": query,
                    "count": count,
                    "safesearch": safesearch,
                    "search_lang": "en"  # Can be made configurable
                }

                logger.info(f"Searching Brave (attempt {attempt + 1}/{max_retries}): {query}")

                with httpx.Client(timeout=15.0) as client:
                    response = client.get(
                        self.base_url,
                        headers=headers,
                        params=params
                    )
                    response.raise_for_status()

                data = response.json()
                results = self._parse_results(data)

                logger.info(f"Found {len(results)} web results")
                return results

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited by Brave API. Retrying in {2 ** attempt} seconds...")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                elif e.response.status_code == 401:  # Unauthorized
                    logger.error("Invalid Brave API key. Check BRAVE_API_KEY in .env")
                    return []
                else:
                    logger.error(f"HTTP {e.response.status_code} error during web search: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return []
            except httpx.TimeoutException as e:
                logger.warning(f"Timeout during web search (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return []
            except httpx.HTTPError as e:
                logger.error(f"HTTP error during web search: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return []
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from Brave API: {e}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error during web search: {e}")
                return []

        logger.error(f"Failed to search after {max_retries} attempts")
        return []

    def _parse_results(self, data: Dict) -> List[Dict]:
        """Parse Brave Search API response"""
        results = []

        web_results = data.get("web", {}).get("results", [])

        for item in web_results:
            result = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
                "published": item.get("page_age", ""),
                "language": item.get("language", "en")
            }
            results.append(result)

        return results

    def search_with_context(
        self,
        query: str,
        count: Optional[int] = None
    ) -> Dict:
        """
        Search and return formatted context for LLM

        Args:
            query: Search query
            count: Number of results

        Returns:
            Dict with formatted context and metadata
        """
        results = self.search(query, count)

        if not results:
            return {
                "context": "",
                "sources": [],
                "count": 0
            }

        # Format context for LLM
        context_parts = []
        sources = []

        for i, result in enumerate(results, 1):
            context_parts.append(f"[Web Result {i}]")
            context_parts.append(f"Title: {result['title']}")
            context_parts.append(f"Content: {result['snippet']}")
            context_parts.append("")

            sources.append({
                "title": result["title"],
                "url": result["url"],
                "rank": i
            })

        context = "\n".join(context_parts)

        return {
            "context": context,
            "sources": sources,
            "count": len(results),
            "results": results
        }


class WebSearchFallback:
    """
    Intelligent web search fallback
    Decides when to use web search based on document relevance
    """

    def __init__(self, brave_search: Optional[BraveSearchMCP] = None):
        self.config = get_config()
        self.brave_search = brave_search or BraveSearchMCP()
        self.similarity_threshold = self.config.leann.similarity_threshold

    def should_search_web(
        self,
        query: str,
        document_results: List[Dict]
    ) -> bool:
        """
        Decide whether to search the web based on document results

        Args:
            query: User query
            document_results: Results from LEANN search

        Returns:
            True if web search should be performed
        """
        if not self.config.web_search.enabled:
            return False

        if not self.brave_search.is_available():
            return False

        # No documents found
        if not document_results:
            logger.info("No documents found - triggering web search")
            return True

        # Check relevance scores
        max_score = max(
            (doc.get("score", 0) for doc in document_results),
            default=0
        )

        if max_score < self.similarity_threshold:
            logger.info(
                f"Low relevance score ({max_score:.2f} < {self.similarity_threshold}) "
                "- triggering web search"
            )
            return True

        # Check if documents are too short/insufficient
        total_text_length = sum(
            len(doc.get("text", "")) for doc in document_results
        )

        if total_text_length < 200:  # Less than 200 chars total
            logger.info("Insufficient document context - triggering web search")
            return True

        return False

    def get_combined_context(
        self,
        query: str,
        document_results: List[Dict],
        force_web_search: bool = False
    ) -> Dict:
        """
        Get combined context from documents and optionally web

        Args:
            query: User query
            document_results: Results from LEANN
            force_web_search: Force web search regardless of document quality

        Returns:
            Dict with combined context and metadata
        """
        # Filter documents by relevance threshold
        # Only include documents that meet the minimum similarity score
        relevant_docs = [
            doc for doc in document_results
            if doc.get("score", 0) >= self.similarity_threshold
        ]

        if len(relevant_docs) < len(document_results):
            filtered_count = len(document_results) - len(relevant_docs)
            logger.info(f"Filtered out {filtered_count} low-relevance documents (threshold: {self.similarity_threshold})")

        context_data = {
            "documents": relevant_docs,  # Only relevant docs
            "web_results": [],
            "used_web_search": False,
            "document_count": len(relevant_docs),
            "web_count": 0
        }

        # Decide whether to search web (based on original results)
        should_search = force_web_search or self.should_search_web(
            query, document_results
        )

        if should_search:
            web_data = self.brave_search.search_with_context(query)
            context_data["web_results"] = web_data.get("results", [])
            context_data["used_web_search"] = True
            context_data["web_count"] = web_data.get("count", 0)

        return context_data


if __name__ == "__main__":
    # Test Brave Search
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    search = BraveSearchMCP()

    if search.is_available():
        # Test search
        results = search.search("Python programming language", count=3)

        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Snippet: {result['snippet'][:100]}...")
            print()

        # Test context formatting
        context_data = search.search_with_context("What is Python?", count=2)
        print("\nFormatted context:")
        print(context_data["context"][:500])
        print(f"\nSources: {len(context_data['sources'])}")

    else:
        print("Brave Search not available. Set BRAVE_API_KEY in .env")

    # Test fallback logic
    fallback = WebSearchFallback(search)

    # Test with low relevance documents
    mock_docs = [
        {"text": "Short text", "score": 0.3},
        {"text": "Another short", "score": 0.4}
    ]

    should_search = fallback.should_search_web("test query", mock_docs)
    print(f"\nShould search web with low scores? {should_search}")

    # Test with high relevance documents
    mock_docs_good = [
        {"text": "This is a much longer and more relevant document" * 10, "score": 0.8},
        {"text": "Another good document with lots of content" * 10, "score": 0.7}
    ]

    should_search = fallback.should_search_web("test query", mock_docs_good)
    print(f"Should search web with high scores? {should_search}")
