"""
TTS (Text-to-Speech) Engine for Jarvis
Uses Coqui XTTS-v2 for multilingual speech synthesis and voice cloning
"""

import logging
from pathlib import Path
from typing import Optional, Union, Iterator
import numpy as np

from core.config import get_config

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    XTTS-v2 integration for text-to-speech
    Supports 17 languages and voice cloning
    """

    def __init__(self):
        self.config = get_config()
        self.tts = None

        if self.config.tts.enabled:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize XTTS-v2 model"""
        try:
            from TTS.api import TTS

            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

            logger.info(f"Loading TTS model: {model_name}")

            self.tts = TTS(
                model_name=model_name,
                gpu=(self.config.tts.device == "cuda")
            )

            logger.info("TTS model loaded successfully")

        except ImportError:
            logger.error(
                "TTS package not installed. "
                "Install with: pip install TTS"
            )
            raise
        except Exception as e:
            logger.error(f"Error loading TTS model: {e}")
            raise

    def is_available(self) -> bool:
        """Check if TTS is available"""
        return self.config.tts.enabled and self.tts is not None

    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
        speaker_wav: Optional[Union[str, Path]] = None
    ) -> Optional[np.ndarray]:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            output_path: Path to save audio file (optional)
            language: Language code (e.g., "en", "es", "fr")
            speaker_wav: Path to reference voice sample for cloning

        Returns:
            Audio array if output_path is None, else None
        """
        if not self.is_available():
            raise RuntimeError("TTS not available. Check configuration.")

        language = language or self.config.tts.language
        speaker_wav = speaker_wav or self.config.tts.speaker_wav

        try:
            logger.info(f"Synthesizing text: {text[:50]}...")

            if output_path:
                # Save to file
                output_path = str(output_path)
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=language,
                    speaker_wav=speaker_wav
                )
                logger.info(f"Audio saved to: {output_path}")
                return None
            else:
                # Return audio array
                audio = self.tts.tts(
                    text=text,
                    language=language,
                    speaker_wav=speaker_wav
                )
                return np.array(audio)

        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise

    def synthesize_streaming(
        self,
        text: str,
        language: Optional[str] = None,
        speaker_wav: Optional[Union[str, Path]] = None,
        chunk_size: int = 200
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech in streaming mode (sentence by sentence)

        Args:
            text: Text to synthesize
            language: Language code
            speaker_wav: Reference voice sample
            chunk_size: Character chunk size for streaming

        Yields:
            Audio chunks as numpy arrays
        """
        if not self.is_available():
            raise RuntimeError("TTS not available. Check configuration.")

        language = language or self.config.tts.language
        speaker_wav = speaker_wav or self.config.tts.speaker_wav

        # Split text into sentences
        sentences = self._split_into_sentences(text)

        logger.info(f"Streaming synthesis for {len(sentences)} sentences")

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            try:
                logger.debug(f"Synthesizing sentence {i+1}/{len(sentences)}")

                audio = self.tts.tts(
                    text=sentence,
                    language=language,
                    speaker_wav=speaker_wav
                )

                yield np.array(audio)

            except Exception as e:
                logger.error(f"Error synthesizing sentence {i+1}: {e}")
                continue

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences for streaming"""
        import re

        # Simple sentence splitting (can be improved with nltk)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def clone_voice(
        self,
        text: str,
        speaker_wav: Union[str, Path],
        output_path: Union[str, Path],
        language: Optional[str] = None
    ) -> None:
        """
        Clone a voice and synthesize text with it

        Args:
            text: Text to synthesize
            speaker_wav: Path to reference voice sample (6+ seconds)
            output_path: Where to save synthesized audio
            language: Target language
        """
        if not self.is_available():
            raise RuntimeError("TTS not available. Check configuration.")

        speaker_wav = str(speaker_wav)
        output_path = str(output_path)
        language = language or self.config.tts.language

        logger.info(f"Cloning voice from: {speaker_wav}")

        try:
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                language=language,
                speaker_wav=speaker_wav
            )

            logger.info(f"Voice cloned and saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error cloning voice: {e}")
            raise

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages"""
        # XTTS-v2 supported languages
        return [
            "en",  # English
            "es",  # Spanish
            "fr",  # French
            "de",  # German
            "it",  # Italian
            "pt",  # Portuguese
            "pl",  # Polish
            "tr",  # Turkish
            "ru",  # Russian
            "nl",  # Dutch
            "cs",  # Czech
            "ar",  # Arabic
            "zh-cn",  # Chinese
            "ja",  # Japanese
            "hu",  # Hungarian
            "ko",  # Korean
            "hi"   # Hindi
        ]

    def validate_speaker_wav(
        self,
        speaker_wav: Union[str, Path]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate speaker WAV file for voice cloning

        Args:
            speaker_wav: Path to speaker audio

        Returns:
            (is_valid, error_message)
        """
        speaker_wav = Path(speaker_wav)

        if not speaker_wav.exists():
            return False, "File not found"

        if not speaker_wav.suffix.lower() in ['.wav', '.mp3', '.flac']:
            return False, "File must be WAV, MP3, or FLAC"

        # Check duration (should be 6+ seconds)
        try:
            import soundfile as sf
            data, samplerate = sf.read(str(speaker_wav))
            duration = len(data) / samplerate

            if duration < 6:
                return False, f"Audio too short ({duration:.1f}s). Need 6+ seconds"

            logger.info(f"Speaker audio validated: {duration:.1f}s")
            return True, None

        except Exception as e:
            return False, f"Error reading audio: {str(e)}"


if __name__ == "__main__":
    # Test TTS engine
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    try:
        tts = TTSEngine()

        if tts.is_available():
            print("✅ TTS engine initialized successfully")
            print(f"Device: {tts.config.tts.device}")
            print(f"Default language: {tts.config.tts.language}")

            # Get supported languages
            langs = tts.get_supported_languages()
            print(f"\n📋 Supported languages: {langs}")

            # Test synthesis
            test_text = "Hello, I am Jarvis, your personal AI assistant."
            output_file = "/tmp/test_tts.wav"

            print(f"\n🔊 Synthesizing: {test_text}")
            tts.synthesize(test_text, output_path=output_file)
            print(f"✅ Audio saved to: {output_file}")

            # Test streaming
            if tts.config.tts.streaming:
                print("\n🔊 Testing streaming synthesis...")
                long_text = (
                    "This is a longer text to test streaming synthesis. "
                    "It will be split into multiple sentences. "
                    "Each sentence will be synthesized separately."
                )

                chunks = list(tts.synthesize_streaming(long_text))
                print(f"✅ Generated {len(chunks)} audio chunks")

        else:
            print("❌ TTS not available. Set TTS_ENABLED=true in .env")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install TTS: pip install TTS")
    except Exception as e:
        print(f"❌ Error: {e}")
