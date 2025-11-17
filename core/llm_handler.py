"""
LLM Handler for Jarvis
Supports DeepSeek, OpenAI, and Ollama with a unified interface
"""

import logging
from typing import List, Dict, Optional, Iterator
from openai import OpenAI
import httpx

from .config import get_config

logger = logging.getLogger(__name__)


class LLMHandler:
    """Unified LLM handler supporting multiple providers"""

    def __init__(self):
        self.config = get_config()
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        provider = self.config.llm.provider

        if provider == "deepseek":
            if not self.config.llm.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY not set in .env")

            self.client = OpenAI(
                api_key=self.config.llm.deepseek_api_key,
                base_url=self.config.llm.deepseek_base_url
            )
            self.model = self.config.llm.deepseek_model
            logger.info(f"Initialized DeepSeek client with model: {self.model}")

        elif provider == "openai":
            if not self.config.llm.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in .env")

            self.client = OpenAI(api_key=self.config.llm.openai_api_key)
            self.model = self.config.llm.openai_model
            logger.info(f"Initialized OpenAI client with model: {self.model}")

        elif provider == "ollama":
            self.client = OpenAI(
                base_url=self.config.llm.ollama_base_url,
                api_key="ollama"  # Ollama doesn't need real API key
            )
            self.model = self.config.llm.ollama_model
            logger.info(f"Initialized Ollama client with model: {self.model}")

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Generate a response from the LLM

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Complete response string or iterator of response chunks
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": stream
            }

            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**kwargs)

            if stream:
                return self._stream_response(response)
            else:
                content = response.choices[0].message.content
                return self._clean_response(content)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _clean_response(self, text: str) -> str:
        """Remove formal phrases from response"""
        # Phrases to remove from the start of responses
        remove_phrases = [
            "Based on the context provided, ",
            "Based on the context, ",
            "Based on the above context, ",
            "Based on the information provided, ",
            "According to the documents, ",
            "According to the document, ",
            "According to the information provided, ",
            "According to the context, ",
            "The context indicates that ",
            "The context shows that ",
            "The documents show that ",
            "The document shows that ",
            "From the context, ",
            "From the documents, ",
            "The information shows that ",
            "The information indicates that ",
            "As per the context, ",
            "As mentioned in the context, ",
        ]

        cleaned = text.strip()

        # Try to remove formal phrases from start
        for phrase in remove_phrases:
            if cleaned.startswith(phrase):
                cleaned = cleaned[len(phrase):].strip()
                # Capitalize first letter after removal
                if cleaned:
                    cleaned = cleaned[0].upper() + cleaned[1:]
                break

        return cleaned

    def _stream_response(self, response) -> Iterator[str]:
        """Stream response chunks"""
        buffer = ""
        first_chunk = True

        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                buffer += content

                # Clean the first part of the response
                if first_chunk and len(buffer) > 50:
                    cleaned = self._clean_response(buffer)
                    yield cleaned
                    buffer = ""
                    first_chunk = False
                elif not first_chunk:
                    yield content

        # Yield any remaining buffer
        if buffer:
            if first_chunk:
                yield self._clean_response(buffer)
            else:
                yield buffer

    def ask(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Ask a question with optional context (RAG)

        Args:
            query: User question
            context: Retrieved context from documents/web
            system_prompt: Custom system prompt
            temperature: Sampling temperature
            stream: Whether to stream the response

        Returns:
            LLM response
        """
        messages = []

        # System prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            default_system = """You are Jarvis, a humble, emotionally intelligent AI assistant.
Speak naturally — like a thoughtful human conversation partner.

Do NOT start responses with phrases like:
- "Based on the context provided"
- "According to the document"
- "As an AI language model"
- "The information shows"
- "From the documents"

Instead, speak directly and naturally, as if you already know the information.

If uncertain, use gentle phrasing such as:
- "It seems like..."
- "From what I understand..."
- "I'm not completely sure, but it might be..."

Always keep the tone:
- Friendly and warm
- Humble and polite
- Respectful to the person you're speaking with
- Concise but helpful
- Natural, like a real conversation

Remember: You're having a conversation, not giving a formal report."""
            messages.append({"role": "system", "content": default_system})

        # Add context if provided
        if context:
            user_message = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            user_message = query

        messages.append({"role": "user", "content": user_message})

        return self.generate_response(messages, temperature=temperature, stream=stream)

    def summarize(self, text: str, max_length: int = 150) -> str:
        """Summarize text"""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text concisely."
            },
            {
                "role": "user",
                "content": f"Summarize the following text in about {max_length} words:\n\n{text}"
            }
        ]
        return self.generate_response(messages, temperature=0.3)

    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract keywords from text"""
        messages = [
            {
                "role": "system",
                "content": "Extract the most important keywords from the text. Return only the keywords separated by commas."
            },
            {
                "role": "user",
                "content": f"Extract {num_keywords} keywords from:\n\n{text}"
            }
        ]
        response = self.generate_response(messages, temperature=0.3)
        keywords = [k.strip() for k in response.split(',')]
        return keywords[:num_keywords]

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite user query for better search results
        Useful for improving RAG retrieval
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer. "
                    "Rewrite the user's question to make it better for semantic search. "
                    "Keep it concise and focused. Return only the rewritten query."
                )
            },
            {
                "role": "user",
                "content": f"Rewrite this query for better search: {query}"
            }
        ]
        return self.generate_response(messages, temperature=0.3)

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Multi-turn chat conversation

        Args:
            messages: Full conversation history
            stream: Whether to stream the response

        Returns:
            Assistant response
        """
        # Ensure system message exists
        if not messages or messages[0]["role"] != "system":
            system_msg = {
                "role": "system",
                "content": """You are Jarvis, a humble and helpful AI assistant.
Speak naturally like a friendly, thoughtful person.
Be polite, respectful, and conversational.
Never use formal phrases like 'Based on the context' or 'According to documents'.
Just chat naturally and warmly."""
            }
            messages.insert(0, system_msg)

        return self.generate_response(messages, stream=stream)


class RAGPromptBuilder:
    """Helper class to build RAG prompts with retrieved context"""

    @staticmethod
    def build_rag_prompt(
        query: str,
        documents: List[Dict],
        web_results: Optional[List[Dict]] = None,
        include_sources: bool = True
    ) -> str:
        """
        Build a comprehensive prompt with document and web context

        Args:
            query: User question
            documents: Retrieved documents from LEANN
            web_results: Retrieved web search results
            include_sources: Whether to include source information

        Returns:
            Formatted prompt with context
        """
        context_parts = []

        # Add document context
        if documents:
            context_parts.append("## Here's what I found in your files:\n")
            for i, doc in enumerate(documents, 1):
                text = doc.get("text", "")
                score = doc.get("score", 0)
                source = doc.get("metadata", {}).get("source", "Unknown")

                context_parts.append(f"\n[{source}]")
                context_parts.append(f"{text}\n")

        # Add web context
        if web_results:
            context_parts.append("\n## Also grabbed this from the web:\n")
            for i, result in enumerate(web_results, 1):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                url = result.get("url", "")

                context_parts.append(f"\n[{title}]")
                context_parts.append(f"{snippet}")
                if include_sources:
                    context_parts.append(f"Source: {url}")
                context_parts.append("")

        # Build final prompt
        if context_parts:
            context = "\n".join(context_parts)
            prompt = f"""{context}

---

Question: {query}

IMPORTANT INSTRUCTIONS:
- Speak naturally and warmly, like a helpful friend
- NEVER use formal phrases like "Based on the context provided", "According to the documents", "The information shows"
- State information directly as if you already knew it
- Be humble, polite, and respectful
- If uncertain, say "I think..." or "It seems like..." instead of stating as absolute fact
- Keep it conversational and friendly"""
        else:
            prompt = query

        return prompt


if __name__ == "__main__":
    # Test LLM handler
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    try:
        llm = LLMHandler()

        # Test simple query
        response = llm.ask("What is the capital of France?")
        print(f"Response: {response}")

        # Test with context
        context = "Paris is the capital and largest city of France."
        response = llm.ask("What is the capital of France?", context=context)
        print(f"\nWith context: {response}")

        # Test streaming
        print("\nStreaming response:")
        for chunk in llm.ask("Tell me a short joke", stream=True):
            print(chunk, end="", flush=True)
        print()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set API keys in .env file")
