"""
ASR (Automatic Speech Recognition) Engine for Jarvis
Uses Omnilingual ASR for multilingual speech recognition (1600+ languages)
"""

import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np

from core.config import get_config

logger = logging.getLogger(__name__)


class ASREngine:
    """
    Omnilingual ASR integration for speech-to-text
    Supports 1600+ languages with high accuracy
    """

    def __init__(self):
        self.config = get_config()
        self.model = None
        self.pipeline = None

        if self.config.asr.enabled:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize Omnilingual ASR model"""
        try:
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

            model_card = self.config.asr.model  # e.g., "omniASR_CTC_1B"

            logger.info(f"Loading ASR model: {model_card}")

            self.pipeline = ASRInferencePipeline(
                model_card=model_card,
                device=self.config.asr.device
            )

            logger.info("ASR model loaded successfully")

        except ImportError:
            logger.error(
                "omnilingual-asr not installed. "
                "Install with: pip install omnilingual-asr"
            )
            raise
        except Exception as e:
            logger.error(f"Error loading ASR model: {e}")
            raise

    def is_available(self) -> bool:
        """Check if ASR is available"""
        return self.config.asr.enabled and self.pipeline is not None

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., "eng_Latn", "spa_Latn")
                     If None, uses config default or auto-detects

        Returns:
            Transcribed text
        """
        if not self.is_available():
            raise RuntimeError("ASR not available. Check configuration.")

        audio_path = str(audio_path)
        language = language or self.config.asr.language

        # Auto-detect language if set to "auto"
        if language == "auto":
            language = None  # Let model auto-detect

        try:
            logger.info(f"Transcribing: {audio_path}")

            transcriptions = self.pipeline.transcribe(
                [audio_path],
                lang=[language] if language else None,
                batch_size=1
            )

            text = transcriptions[0] if transcriptions else ""

            logger.info(f"Transcription: {text[:100]}...")
            return text

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    def transcribe_audio_data(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio numpy array to text

        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate of audio
            language: Language code

        Returns:
            Transcribed text
        """
        if not self.is_available():
            raise RuntimeError("ASR not available. Check configuration.")

        language = language or self.config.asr.language

        if language == "auto":
            language = None

        try:
            audio_dict = {
                "waveform": audio_data,
                "sample_rate": sample_rate
            }

            transcriptions = self.pipeline.transcribe(
                [audio_dict],
                lang=[language] if language else None,
                batch_size=1
            )

            text = transcriptions[0] if transcriptions else ""

            logger.info(f"Transcription: {text[:100]}...")
            return text

        except Exception as e:
            logger.error(f"Error transcribing audio data: {e}")
            raise

    def transcribe_batch(
        self,
        audio_paths: list[Union[str, Path]],
        languages: Optional[list[str]] = None,
        batch_size: int = 4
    ) -> list[str]:
        """
        Transcribe multiple audio files in batch

        Args:
            audio_paths: List of audio file paths
            languages: List of language codes (one per file)
            batch_size: Batch size for processing

        Returns:
            List of transcribed texts
        """
        if not self.is_available():
            raise RuntimeError("ASR not available. Check configuration.")

        audio_paths = [str(p) for p in audio_paths]

        # Handle languages
        if languages:
            if len(languages) != len(audio_paths):
                raise ValueError("languages list must match audio_paths length")
        else:
            default_lang = self.config.asr.language
            if default_lang == "auto":
                languages = None
            else:
                languages = [default_lang] * len(audio_paths)

        try:
            logger.info(f"Transcribing batch of {len(audio_paths)} files")

            transcriptions = self.pipeline.transcribe(
                audio_paths,
                lang=languages,
                batch_size=batch_size
            )

            logger.info(f"Transcribed {len(transcriptions)} audio files")
            return transcriptions

        except Exception as e:
            logger.error(f"Error transcribing batch: {e}")
            raise

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages"""
        try:
            from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
            return list(supported_langs)
        except ImportError:
            logger.warning("Cannot import supported languages")
            return []

    def detect_language(
        self,
        audio_path: Union[str, Path]
    ) -> str:
        """
        Detect language from audio (if supported by model)

        Args:
            audio_path: Path to audio file

        Returns:
            Detected language code
        """
        # Note: Language detection may require transcribing first
        # This is a placeholder for future language detection feature
        logger.warning("Language detection not yet implemented")
        return "unknown"


if __name__ == "__main__":
    # Test ASR engine
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    try:
        asr = ASREngine()

        if asr.is_available():
            print("✅ ASR engine initialized successfully")
            print(f"Model: {asr.config.asr.model}")
            print(f"Device: {asr.config.asr.device}")
            print(f"Default language: {asr.config.asr.language}")

            # Get supported languages
            langs = asr.get_supported_languages()
            print(f"\n📋 Supported languages: {len(langs)}")
            if langs:
                print(f"Examples: {langs[:10]}")

            # Test with audio file (if provided)
            import sys
            if len(sys.argv) > 1:
                audio_file = sys.argv[1]
                print(f"\n🎤 Transcribing: {audio_file}")
                text = asr.transcribe_file(audio_file)
                print(f"📝 Result: {text}")
        else:
            print("❌ ASR not available. Set ASR_ENABLED=true in .env")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install omnilingual-asr: pip install omnilingual-asr")
    except Exception as e:
        print(f"❌ Error: {e}")
