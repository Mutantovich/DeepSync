import torch
import torchaudio
import logging
import sys
import os
import gc
import re
from demucs.apply import apply_model
from demucs.pretrained import get_model
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from tqdm import tqdm
from llama_cpp import Llama

# ----------------------------------------------------------------------
# Отключаем телеметрию и принудительно включаем офлайн-режим
# ----------------------------------------------------------------------
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_HUB_OFFLINE"] = "1"

# ----------------------------------------------------------------------
# Настройка логирования
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_stereo(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Преобразует аудио в стерео (2 канала) для Demucs."""
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
        logger.info("Аудио преобразовано из моно в стерео (дублирование канала)")
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]
        logger.info(f"Аудио имело {waveform.shape[0]} каналов, использованы первые два")
    else:
        logger.info("Аудио уже стерео (2 канала)")
    return waveform


class AudioProcessor:
    CACHE_DIR = os.path.expanduser("~/.cache/huggingface/")
    MODELS = {
        "whisper-base": "openai/whisper-base",
    }

    # Параметры модели OmniLing на Hugging Face Hub (GGUF)
    OMNILING_REPO = "QuantFactory/OmniLing-V1-8b-experimental-GGUF"
    OMNILING_FILENAME = "OmniLing-V1-8b-experimental.Q4_K_M.gguf"

    def __init__(self):
        logger.info("Начинаем инициализацию AudioProcessor...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {self.device}")

        os.makedirs(self.CACHE_DIR, exist_ok=True)

        self._demucs_model = None

        # Проверяем и загружаем Whisper
        self._ensure_models_available()
        self._initialize_whisper()

        # Загружаем модель перевода OmniLing через llama-cpp-python (автоматическая загрузка)
        logger.info("Инициализация модели перевода OmniLing (GGUF)...")
        self._initialize_omnilig()

        logger.info("Инициализация AudioProcessor завершена успешно")

    # ------------------------------------------------------------------
    # Работа с кэшем Whisper
    # ------------------------------------------------------------------
    def _model_is_cached(self, model_id: str) -> bool:
        try:
            WhisperProcessor.from_pretrained(model_id, local_files_only=True, cache_dir=self.CACHE_DIR)
            return True
        except Exception:
            return False

    def _ensure_models_available(self):
        model_name = "whisper-base"
        model_id = self.MODELS[model_name]
        if self._model_is_cached(model_id):
            logger.info(f"Модель {model_name} уже есть в кэше")
        else:
            logger.info(f"Модель {model_name} не найдена, загружаем...")
            self._download_model(model_name, model_id)

    def _download_model(self, model_name: str, model_id: str):
        from huggingface_hub import snapshot_download
        import shutil

        logger.info(f"Загрузка {model_name} ({model_id})...")
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=self.CACHE_DIR,
                tqdm_class=tqdm,
                force_download=False,
                resume_download=True,
            )
            logger.info(f"Модель {model_name} успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки {model_name}: {e}")
            cache_subdir = "models--" + model_id.replace("/", "--")
            cache_path = os.path.join(self.CACHE_DIR, cache_subdir)
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path, ignore_errors=True)
            raise RuntimeError(f"Не удалось загрузить {model_name}") from e

    # ------------------------------------------------------------------
    # Инициализация Whisper
    # ------------------------------------------------------------------
    def _initialize_whisper(self):
        whisper_id = self.MODELS["whisper-base"]
        try:
            self.whisper_processor = WhisperProcessor.from_pretrained(
                whisper_id, local_files_only=True, cache_dir=self.CACHE_DIR
            )
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                whisper_id, local_files_only=True, cache_dir=self.CACHE_DIR
            ).to(self.device)
            logger.info("Whisper загружен из кэша")
        except Exception:
            logger.info("Локальная копия Whisper не найдена, загружаем из сети...")
            self.whisper_processor = WhisperProcessor.from_pretrained(
                whisper_id, local_files_only=False, cache_dir=self.CACHE_DIR
            )
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                whisper_id, local_files_only=False, cache_dir=self.CACHE_DIR
            ).to(self.device)
            logger.info("Whisper загружен из сети и сохранён в кэш")
        logger.info("Whisper готов")

    # ------------------------------------------------------------------
    # Инициализация OmniLing через llama-cpp-python (автоматическая загрузка)
    # ------------------------------------------------------------------
    def _initialize_omnilig(self):
        logger.info(f"Загрузка модели OmniLing из репозитория {self.OMNILING_REPO}, файл {self.OMNILING_FILENAME}")
        try:
            self.translator = Llama.from_pretrained(
                repo_id=self.OMNILING_REPO,
                filename=self.OMNILING_FILENAME,
                n_gpu_layers=-1,        # все слои на GPU (если не хватит памяти, уменьшите)
                n_ctx=2048,
                verbose=False,
            )
            self.translator_tokenizer = None  # не нужен, всё встроено
            logger.info("Модель перевода OmniLing успешно загружена с Hugging Face Hub")
        except Exception as e:
            logger.error(f"Ошибка загрузки OmniLing: {e}", exc_info=True)
            raise RuntimeError("Не удалось загрузить модель перевода OmniLing") from e

    # ------------------------------------------------------------------
    # Ленивая загрузка Demucs
    # ------------------------------------------------------------------
    def _get_demucs(self):
        if self._demucs_model is None:
            logger.info("Ленивая загрузка Demucs (htdemucs)...")
            try:
                self._demucs_model = get_model('htdemucs')
                self._demucs_model.eval()
                self._demucs_model.to(self.device)
                logger.info("Demucs успешно загружен")
            except Exception as e:
                logger.error(f"Ошибка загрузки Demucs: {e}", exc_info=True)
                raise RuntimeError("Не удалось загрузить модель Demucs") from e
        return self._demucs_model

    # ------------------------------------------------------------------
    # Основные методы обработки
    # ------------------------------------------------------------------
    def separate_voice_background(self, audio_path: str) -> tuple[str, str]:
        logger.info(f"Разделение аудио: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        waveform = ensure_stereo(waveform, sample_rate)
        demucs_model = self._get_demucs()

        with torch.no_grad():
            sources = apply_model(
                demucs_model,
                waveform.unsqueeze(0),
                device=self.device,
                shifts=1,
                split=True,
                overlap=0.25,
                progress=True,
            )[0]

        vocals = sources[3].cpu()
        background = (sources[0] + sources[1] + sources[2]).cpu()

        output_dir = os.path.dirname(audio_path)
        vocals_path = os.path.join(output_dir, "vocals.wav")
        background_path = os.path.join(output_dir, "background.wav")

        torchaudio.save(vocals_path, vocals, sample_rate)
        torchaudio.save(background_path, background, sample_rate)

        logger.info(f"Сохранено: {vocals_path} и {background_path}")
        return vocals_path, background_path

    def transcribe_audio(self, audio_path: str) -> str:
        logger.info(f"Транскрибация: {audio_path}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        waveform, sample_rate = torchaudio.load(audio_path)
        target_sr = 16000
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform.squeeze(0)

        segment_length = 30 * sample_rate
        overlap = 1 * sample_rate
        stride = segment_length - overlap
        total_samples = waveform.shape[0]

        if total_samples <= segment_length:
            segments = [(0, total_samples)]
        else:
            segments = []
            start = 0
            while start < total_samples:
                end = min(start + segment_length, total_samples)
                segments.append((start, end))
                if end == total_samples:
                    break
                start += stride
            logger.info(f"Аудио разбито на {len(segments)} сегментов")

        transcriptions = []
        for i, (start, end) in enumerate(segments, 1):
            segment = waveform[start:end]
            if segment.shape[0] < 1 * sample_rate:
                continue
            input_features = self.whisper_processor(
                segment.cpu(), sampling_rate=sample_rate, return_tensors="pt"
            ).input_features.to(self.device)
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(input_features)
            segment_text = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcriptions.append(segment_text)
            logger.info(f"Сегмент {i}/{len(segments)}: '{segment_text[:50]}...'")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        transcription = " ".join(transcriptions).strip()
        logger.info(f"Транскрибация завершена. Итог: {transcription[:60]}...")
        return transcription

    def translate_text(self, text: str) -> str:
        """Перевод с английского на русский с помощью OmniLing (GGUF)."""
        logger.info("Перевод текста...")
        if not text or len(text.strip()) == 0:
            return ""

        # Разбиваем на предложения и группируем в чанки по 100 слов
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) == 1 and not any(ch in text for ch in ['.', '!', '?']):
            sentences = [text]

        max_words_per_chunk = 100
        chunks = []
        current_chunk = []
        current_len = 0
        for sent in sentences:
            word_count = len(sent.split())
            if word_count == 0:
                continue
            if current_len + word_count <= max_words_per_chunk:
                current_chunk.append(sent)
                current_len += word_count
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sent]
                current_len = word_count
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        logger.info(f"Текст разбит на {len(chunks)} чанков для перевода")

        translated_chunks = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Перевод чанка {i}/{len(chunks)} (символов: {len(chunk)}, слов: {len(chunk.split())})")
            if not chunk.strip():
                translated_chunks.append("")
                continue

            # Промпт в формате Llama 3 Instruct (ожидаемый моделью OmniLing)
            prompt = (
                f"<|start_header_id|>system<|end_header_id|>\n"
                f"You are a helpful AI assistant.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"Translate this text from English to Russian:\n\n{chunk}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )

            try:
                response = self.translator(
                    prompt,
                    max_tokens=512,
                    temperature=0.2,
                    stop=["<|eot_id|>"],
                    echo=False,
                )
                translation = response['choices'][0]['text'].strip()
                translated_chunks.append(translation)
                logger.info(f"Чанк {i} переведён")
            except Exception as e:
                logger.error(f"Ошибка при переводе чанка {i}: {e}", exc_info=True)
                translated_chunks.append("")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        result = " ".join(translated_chunks).strip()
        logger.info(f"Перевод завершён. Результат: {result[:60]}...")
        return result
