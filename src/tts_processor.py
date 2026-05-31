import torch
import torchaudio
from TTS.api import TTS
import os
import logging
import re
from pydub import AudioSegment
import io

logger = logging.getLogger(__name__)

class TTSProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TTS инициализация на устройстве: {self.device}")
        # Используем Coqui TTS с моделью XTTS v2
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(self.device)
        # Максимальное количество символов на один запрос (безопасный запас)
        self.max_chars_per_chunk = 2000  # ~400 токенов XTTS

    def clone_voice(self, reference_audio: str, save_path: str) -> str:
        """Клонирует голос из референсного аудио"""
        self.reference_audio = reference_audio
        logger.info(f"Голос клонирован из {reference_audio}")
        return reference_audio

    def _split_text_into_chunks(self, text: str) -> list:
        """
        Разбивает текст на фрагменты, не превышающие max_chars_per_chunk.
        Старается разбивать по границам предложений.
        """
        # Если текст короткий, возвращаем как есть
        if len(text) <= self.max_chars_per_chunk:
            return [text]

        # Разбиваем по предложениям (.!?;:)
        sentences = re.split(r'(?<=[.!?;:])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Если предложение слишком длинное (например, нет знака препинания), разбиваем по словам
            if len(sentence) > self.max_chars_per_chunk:
                words = sentence.split()
                temp = ""
                for word in words:
                    if len(temp) + len(word) + 1 <= self.max_chars_per_chunk:
                        temp += (word + " ")
                    else:
                        if temp:
                            chunks.append(temp.strip())
                        temp = word + " "
                if temp:
                    chunks.append(temp.strip())
                continue

            # Обычный случай: добавляем предложение к текущему фрагменту
            if len(current_chunk) + len(sentence) + 1 <= self.max_chars_per_chunk:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info(f"Текст разбит на {len(chunks)} фрагментов для TTS")
        return chunks

    def generate_speech(self, text: str, output_path: str) -> str:
        """Генерирует речь с клонированным голосом, поддерживая длинные тексты."""
        logger.info("Генерация речи с помощью XTTS...")
        chunks = self._split_text_into_chunks(text)

        if not chunks:
            logger.warning("Пустой текст для генерации речи")
            return ""

        # Если только один фрагмент, генерируем напрямую
        if len(chunks) == 1:
            self.tts.tts_to_file(
                text=chunks[0],
                file_path=output_path,
                speaker_wav=self.reference_audio,
                language="ru"
            )
            logger.info(f"Речь сгенерирована в {output_path}")
            return output_path

        # Генерируем каждый фрагмент во временный файл, затем объединяем
        temp_files = []
        for i, chunk in enumerate(chunks):
            temp_path = f"{output_path}.part{i}.wav"
            logger.info(f"Генерация фрагмента {i+1}/{len(chunks)} (длина: {len(chunk)} символов)")
            try:
                self.tts.tts_to_file(
                    text=chunk,
                    file_path=temp_path,
                    speaker_wav=self.reference_audio,
                    language="ru"
                )
                temp_files.append(temp_path)
            except Exception as e:
                logger.error(f"Ошибка при генерации фрагмента {i+1}: {e}")
                # Пропускаем проблемный фрагмент? Но лучше прервать, так как результат будет неполным
                raise

        # Объединяем все временные аудиофайлы
        logger.info("Объединение аудиофрагментов...")
        combined = AudioSegment.empty()
        for temp_path in temp_files:
            segment = AudioSegment.from_wav(temp_path)
            combined += segment
            # Удаляем временный файл
            os.remove(temp_path)

        # Сохраняем результат
        combined.export(output_path, format="wav")
        logger.info(f"Речь сгенерирована и объединена в {output_path}")
        return output_path
