"""
RAG Engine for Jarvis
Integrates LEANN, LLM, and web search for intelligent question answering
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from leann import LeannBuilder, LeannSearcher, LeannChat
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

from .config import get_config
from .user_manager import UserManager
from .llm_handler import LLMHandler, RAGPromptBuilder
from .mcp_search import BraveSearchMCP, WebSearchFallback

logger = logging.getLogger(__name__)


class RAGEngine:
    """Main RAG engine integrating all components"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.config = get_config()
        self.user_manager = UserManager()
        self.llm = LLMHandler()
        self.brave_search = BraveSearchMCP()
        self.web_fallback = WebSearchFallback(self.brave_search)

        # Ensure user exists
        self.user_manager.create_user(user_id)

        # LEANN components
        self.searcher: Optional[LeannSearcher] = None
        self._load_searcher()

    def _load_searcher(self):
        """Load LEANN searcher if index exists"""
        if self.user_manager.has_index(self.user_id):
            index_path = str(self.user_manager.get_user_index_path(self.user_id))
            try:
                # Verify meta file exists before loading searcher
                from pathlib import Path
                import shutil

                leann_meta = Path(f"{index_path}.leann.meta.json")
                expected_meta = Path(f"{index_path}.meta.json")

                # Ensure expected meta file exists
                if not expected_meta.exists() and leann_meta.exists():
                    shutil.copy2(leann_meta, expected_meta)
                    logger.info(f"Created missing meta file for searcher")

                self.searcher = LeannSearcher(index_path)
                logger.info(f"Loaded LEANN index for user {self.user_id}")
            except FileNotFoundError as e:
                logger.error(f"Index files missing for user {self.user_id}: {e}. Try rebuilding the index.")
                self.searcher = None
            except Exception as e:
                logger.error(f"Error loading LEANN index: {e}. Index may be corrupted - try rebuilding.")
                self.searcher = None
        else:
            logger.info(f"No index found for user {self.user_id}")

    def build_index(self, force_rebuild: bool = False) -> Dict:
        """
        Build LEANN index from user's documents

        Args:
            force_rebuild: Force rebuild even if index exists

        Returns:
            Dict with build status and stats
        """
        index_path = str(self.user_manager.get_user_index_path(self.user_id))

        # Check metadata flag, not physical files (dynamic rebuild when docs change)
        metadata = self.user_manager.get_user_metadata(self.user_id)
        if metadata.get("index_built", False) and not force_rebuild:
            return {
                "status": "exists",
                "message": "Index already exists. Use force_rebuild=True to rebuild."
            }

        # Get user documents
        documents = self.user_manager.get_user_documents(self.user_id)

        if not documents:
            return {
                "status": "error",
                "message": "No documents found. Please upload documents first."
            }

        logger.info(f"Building index for {len(documents)} documents...")

        try:
            # Load and parse documents
            reader = SimpleDirectoryReader(
                input_files=[str(doc) for doc in documents],
                filename_as_id=True
            )
            docs = reader.load_data()

            # Chunk documents
            parser = SentenceSplitter(
                chunk_size=self.config.leann.chunk_size,
                chunk_overlap=self.config.leann.chunk_overlap
            )
            nodes = parser.get_nodes_from_documents(docs)

            logger.info(f"Created {len(nodes)} chunks from documents")

            # Build LEANN index
            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model=self.config.leann.embedding_model,
                embedding_mode="sentence-transformers",
                is_compact=True,
                is_recompute=True
            )

            for node in nodes:
                metadata = {
                    "source": node.metadata.get("file_name", "unknown"),
                    "node_id": node.node_id
                }
                builder.add_text(node.text, metadata=metadata)

            # Delete old index if exists
            if self.user_manager.has_index(self.user_id):
                self.user_manager.delete_index(self.user_id)

            builder.build_index(index_path)

            # LEANN creates .leann.meta.json but searcher expects .meta.json
            # Handle meta file compatibility with robust error recovery
            import shutil
            leann_meta = Path(f"{index_path}.leann.meta.json")
            expected_meta = Path(f"{index_path}.meta.json")

            try:
                if leann_meta.exists():
                    # Copy meta file for searcher compatibility
                    shutil.copy2(leann_meta, expected_meta)
                    logger.info(f"Copied meta file for searcher compatibility: {expected_meta.name}")
                elif expected_meta.exists():
                    # If expected meta already exists, use it
                    logger.info(f"Using existing meta file: {expected_meta.name}")
                else:
                    # Try to find any meta file and use it
                    user_dir = Path(index_path).parent
                    meta_files = list(user_dir.glob(f"{Path(index_path).name}*.meta.json"))
                    if meta_files:
                        shutil.copy2(meta_files[0], expected_meta)
                        logger.warning(f"Copied alternative meta file: {meta_files[0].name} -> {expected_meta.name}")
                    else:
                        logger.warning("No meta file found - searcher may fail to load. Index will need rebuild.")
            except Exception as meta_error:
                logger.error(f"Error handling meta file: {meta_error}. Index may fail to load.")
                # Don't fail the build - let searcher handle missing meta

            # Mark index as built
            self.user_manager.mark_index_built(self.user_id)

            # Reload searcher
            self._load_searcher()

            stats = self.user_manager.get_user_stats(self.user_id)

            logger.info(f"Index built successfully for user {self.user_id}")

            return {
                "status": "success",
                "message": f"Index built with {len(nodes)} chunks",
                "chunk_count": len(nodes),
                "document_count": len(documents),
                "index_size_mb": stats["index_size_mb"],
                "storage_savings_percent": stats["storage_savings_percent"]
            }

        except Exception as e:
            logger.error(f"Error building index: {e}")
            return {
                "status": "error",
                "message": f"Failed to build index: {str(e)}"
            }

    def search_documents(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Search user's documents using LEANN

        Args:
            query: Search query
            top_k: Number of results (default from config)

        Returns:
            List of search results
        """
        if not self.searcher:
            logger.warning(f"No index available for user {self.user_id}")
            return []

        top_k = top_k or self.config.leann.top_k

        try:
            results = self.searcher.search(query, top_k=top_k)

            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append({
                    "text": result.text,
                    "score": result.score,
                    "metadata": result.metadata,
                    "rank": i + 1
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def _is_greeting_or_simple(self, query: str) -> bool:
        """Check if query is a simple greeting or very short query"""
        query_lower = query.lower().strip()
        greetings = ['hi', 'hello', 'hey', 'hai', 'yo', 'sup', 'greetings', 'good morning', 'good evening']

        # Check if it's a greeting
        if query_lower in greetings:
            return True

        # Check if query is too short to be meaningful (< 3 chars)
        # But allow questions with "who", "what", "where", etc even if short
        question_words = ['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose']
        has_question_word = any(word in query_lower.split() for word in question_words)

        if len(query.strip()) < 3:
            return True

        # Don't skip if it's a question, even if short
        if has_question_word:
            return False

        # Skip if just a single word that's not a question
        if len(query.split()) == 1:
            return True

        return False

    def ask(
        self,
        query: str,
        use_web_fallback: bool = True,
        force_web_search: bool = False,
        stream: bool = False
    ) -> Dict | Iterator[Dict]:
        """
        Ask a question with RAG

        Args:
            query: User question
            use_web_fallback: Enable automatic web search fallback
            force_web_search: Force web search even with good document results
            stream: Stream the response

        Returns:
            Dict with answer and metadata, or iterator of response chunks
        """
        # Skip RAG for simple greetings/short queries - just use LLM directly
        if self._is_greeting_or_simple(query) and not force_web_search:
            logger.info(f"Detected greeting/simple query, skipping RAG")
            context_data = {
                "documents": [],
                "web_results": [],
                "used_web_search": False,
                "document_count": 0,
                "web_count": 0
            }

            if stream:
                return self._stream_ask(query, "", context_data)
            else:
                answer = self.llm.ask(query, stream=False)
                return {
                    "answer": answer,
                    "sources": [],
                    "used_web_search": False,
                    "document_count": 0,
                    "web_count": 0
                }

        # Search documents
        doc_results = self.search_documents(query)

        # Get combined context (documents + optional web)
        if use_web_fallback:
            context_data = self.web_fallback.get_combined_context(
                query,
                doc_results,
                force_web_search=force_web_search
            )
        else:
            context_data = {
                "documents": doc_results,
                "web_results": [],
                "used_web_search": False,
                "document_count": len(doc_results),
                "web_count": 0
            }

        # Build prompt with context
        prompt = RAGPromptBuilder.build_rag_prompt(
            query,
            context_data["documents"],
            context_data.get("web_results")
        )

        # Generate response
        try:
            if stream:
                return self._stream_ask(query, prompt, context_data)
            else:
                response = self.llm.ask(
                    query,
                    context=prompt if doc_results or context_data["web_results"] else None
                )

                return {
                    "answer": response,
                    "sources": self._format_sources(context_data),
                    "used_web_search": context_data["used_web_search"],
                    "document_count": context_data["document_count"],
                    "web_count": context_data["web_count"]
                }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "used_web_search": False,
                "document_count": 0,
                "web_count": 0
            }

    def _stream_ask(
        self,
        query: str,
        prompt: str,
        context_data: Dict
    ) -> Iterator[Dict]:
        """Stream response chunks"""
        try:
            # First yield metadata
            yield {
                "type": "metadata",
                "sources": self._format_sources(context_data),
                "used_web_search": context_data["used_web_search"],
                "document_count": context_data["document_count"],
                "web_count": context_data["web_count"]
            }

            # Then stream response
            for chunk in self.llm.ask(query, context=prompt, stream=True):
                yield {
                    "type": "chunk",
                    "content": chunk
                }

        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            yield {
                "type": "error",
                "message": str(e)
            }

    def _format_sources(self, context_data: Dict) -> List[Dict]:
        """Format sources from context data"""
        sources = []

        # Document sources
        for doc in context_data.get("documents", []):
            sources.append({
                "type": "document",
                "title": doc["metadata"].get("source", "Unknown"),
                "score": doc["score"],
                "preview": doc["text"][:200] + "..."
            })

        # Web sources
        for result in context_data.get("web_results", []):
            sources.append({
                "type": "web",
                "title": result["title"],
                "url": result["url"],
                "preview": result["snippet"]
            })

        return sources

    def get_stats(self) -> Dict:
        """Get user statistics"""
        return self.user_manager.get_user_stats(self.user_id)


if __name__ == "__main__":
    # Test RAG engine
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Test with a user
    test_user = "test_user"
    rag = RAGEngine(test_user)

    # Check status
    stats = rag.get_stats()
    print(f"\nUser Stats:")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Index built: {stats['index_built']}")

    if stats['index_built']:
        # Test search
        print("\n--- Testing document search ---")
        results = rag.search_documents("test query", top_k=3)
        print(f"Found {len(results)} results")

        # Test question
        print("\n--- Testing RAG question ---")
        response = rag.ask("What is this about?")
        print(f"Answer: {response['answer'][:200]}...")
        print(f"Used web search: {response['used_web_search']}")
        print(f"Sources: {len(response['sources'])}")
    else:
        print("\n⚠️  No index built. Upload documents and build index first.")
