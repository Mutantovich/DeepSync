import torch
import torchaudio
import logging
import sys
import os
from pathlib import Path
from demucs.pretrained import get_model
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                         M2M100ForConditionalGeneration, M2M100Tokenizer)
from huggingface_hub import snapshot_download
import shutil
from tqdm import tqdm

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessor:
    CACHE_DIR = os.path.expanduser("~/.cache/huggingface/")
    MODELS = {
        "whisper-base": "openai/whisper-base",
        "m2m100": "facebook/m2m100_418M"
    }

    def __init__(self):
        logger.info("Начинаем инициализацию AudioProcessor...")
        
        # Устанавливаем устройство
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {self.device}")
        
        # Проверяем наличие кэш-директории
        logger.info(f"Проверка кэш-директории: {self.CACHE_DIR}")
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        # Убеждаемся, что все модели загружены в кэш
        self._ensure_models_available()
        
        # Инициализируем модели
        logger.info("Инициализация моделей...")
        self._initialize_models()
        
        logger.info("Инициализация AudioProcessor завершена успешно")

    def _model_is_cached(self, model_id: str) -> bool:
        """
        Проверяет, загружена ли модель в кэш, пытаясь загрузить процессор с local_files_only=True.
        Для M2M100 используем токенизатор.
        """
        try:
            if "whisper" in model_id:
                WhisperProcessor.from_pretrained(model_id, local_files_only=True, cache_dir=self.CACHE_DIR)
            else:
                M2M100Tokenizer.from_pretrained(model_id, local_files_only=True, cache_dir=self.CACHE_DIR)
            return True
        except Exception:
            return False

    def _ensure_models_available(self):
        """Загружает все модели, если их нет в кэше"""
        for model_name, model_id in self.MODELS.items():
            if self._model_is_cached(model_id):
                logger.info(f"Модель {model_name} уже есть в кэше")
            else:
                logger.info(f"Модель {model_name} не найдена в кэше, начинаем загрузку...")
                self._download_model(model_name, model_id)

    def _download_model(self, model_name: str, model_id: str):
        """Загружает модель через snapshot_download в стандартную структуру кэша"""
        logger.info(f"Загрузка модели {model_name} ({model_id}) из Hugging Face...")
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=self.CACHE_DIR,
                tqdm_class=tqdm,
                force_download=False,
                resume_download=True
            )
            logger.info(f"Модель {model_name} успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке {model_name}: {str(e)}")
            # Удаляем возможный частичный кэш, чтобы избежать повреждённых данных
            cache_subdir = "models--" + model_id.replace("/", "--")
            cache_path = os.path.join(self.CACHE_DIR, cache_subdir)
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path, ignore_errors=True)
            raise RuntimeError(f"Не удалось загрузить модель {model_name}") from e

    def _initialize_models(self):
        """Инициализирует все модели (Demucs, Whisper, переводчик)"""
        # Инициализация Demucs
        logger.info("Инициализация Demucs...")
        try:
            self.demucs_model = get_model('htdemucs')
            self.demucs_model.eval()
            self.demucs_model.to(self.device)
            logger.info("Demucs инициализирован успешно")
        except Exception as e:
            logger.error(f"Ошибка при инициализации Demucs: {str(e)}")
            raise

        # Инициализация Whisper
        logger.info("Инициализация Whisper...")
        whisper_id = self.MODELS["whisper-base"]
        try:
            # Пытаемся загрузить локально
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
        logger.info("Whisper инициализирован успешно")

        # Инициализация переводчика M2M100
        logger.info("Инициализация переводчика...")
        m2m_id = self.MODELS["m2m100"]
        try:
            self.translator_model = M2M100ForConditionalGeneration.from_pretrained(
                m2m_id, local_files_only=True, cache_dir=self.CACHE_DIR
            ).to(self.device)
            self.translator_tokenizer = M2M100Tokenizer.from_pretrained(
                m2m_id, local_files_only=True, cache_dir=self.CACHE_DIR
            )
            logger.info("Переводчик загружен из кэша")
        except Exception:
            logger.info("Локальная копия переводчика не найдена, загружаем из сети...")
            self.translator_model = M2M100ForConditionalGeneration.from_pretrained(
                m2m_id, local_files_only=False, cache_dir=self.CACHE_DIR
            ).to(self.device)
            self.translator_tokenizer = M2M100Tokenizer.from_pretrained(
                m2m_id, local_files_only=False, cache_dir=self.CACHE_DIR
            )
            logger.info("Переводчик загружен из сети и сохранён в кэш")
        logger.info("Переводчик инициализирован успешно")

    def separate_voice_background(self, audio_path: str) -> tuple[str, str]:
        """Разделяет аудио на голос и фоновые звуки"""
        logger.info(f"Начинаем разделение аудио: {audio_path}")
        
        # Загружаем аудио
        logger.info("Загрузка аудио файла...")
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        logger.info(f"Аудио загружено: {waveform.shape}, sample_rate: {sample_rate}")
        
        # Применяем Demucs
        logger.info("Применяем Demucs для разделения...")
        sources = self.demucs_model.separate(waveform)
        logger.info("Разделение выполнено успешно")
        
        # Получаем директорию для сохранения
        output_dir = os.path.dirname(audio_path)
        vocals_path = os.path.join(output_dir, "vocals.wav")
        background_path = os.path.join(output_dir, "background.wav")
        
        # Сохраняем результаты
        logger.info("Сохранение результатов разделения...")
        torchaudio.save(vocals_path, sources[0].cpu(), sample_rate)
        torchaudio.save(background_path, sources[1].cpu(), sample_rate)
        logger.info("Результаты сохранены успешно")
        
        return vocals_path, background_path
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Транскрибирует аудио в текст"""
        logger.info(f"Начинаем транскрибацию: {audio_path}")
        
        # Загружаем аудио
        logger.info("Загрузка аудио для транскрибации...")
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        logger.info("Аудио загружено успешно")
        
        # Подготавливаем входные данные
        logger.info("Подготовка входных данных для Whisper...")
        input_features = self.whisper_processor(
            waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)
        logger.info("Входные данные подготовлены")
        
        # Получаем транскрипцию
        logger.info("Выполнение транскрибации...")
        predicted_ids = self.whisper_model.generate(input_features)
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.info(f"Транскрибация завершена: {transcription[:50]}...")
        
        return transcription
    
    def translate_text(self, text: str) -> str:
        """Переводит текст с английского на русский"""
        logger.info("Начинаем перевод текста...")
        
        # Токенизируем текст
        logger.info("Токенизация текста...")
        encoded = self.translator_tokenizer(
            text, 
            return_tensors="pt",
            src_lang="en",
            padding=True
        ).to(self.device)
        logger.info("Токенизация завершена")
        
        # Получаем перевод
        logger.info("Выполнение перевода...")
        translated = self.translator_model.generate(
            **encoded,
            forced_bos_token_id=self.translator_tokenizer.get_lang_id("ru")
        )
        translation = self.translator_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        logger.info(f"Перевод завершен: {translation[:50]}...")
        
        return translation
