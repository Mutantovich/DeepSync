import sys
import os
import logging
import traceback
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QPushButton, QFileDialog, QLabel, QProgressBar,
                            QTextEdit, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import signal

# ----------------------------------------------------------------------
# Настройка файлового логирования
# ----------------------------------------------------------------------
def setup_file_logging():
    """Создаёт папку logs и файл лога с меткой времени, возвращает путь к файлу."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"deepdub_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    return log_file

# ----------------------------------------------------------------------
# Обработчик логов для GUI (вывод в QTextEdit)
# ----------------------------------------------------------------------
class QTextEditLogger(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.text_widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.append(msg)

# ----------------------------------------------------------------------
# Поток обработки видео (принимает готовые процессоры)
# ----------------------------------------------------------------------
class ProcessingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, video_path: str, log_file_path: str, audio_processor):
        super().__init__()
        self.video_path = video_path
        self.log_file_path = log_file_path
        self.audio_processor = audio_processor   # уже создан в основном потоке
        self._is_running = True
        
    def _log_error_to_file(self, error_msg, traceback_str):
        """Записывает ошибку и traceback в файл лога."""
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n[ERROR] {error_msg}\n")
            f.write(traceback_str)
            f.write("\n")
    
    def run(self):
        try:
            if not self._is_running:
                return
                
            self.progress.emit("Загрузка компонентов...")
            
            # Импортируем процессоры, которые не требуют тяжёлых моделей (TTS, VideoProcessor)
            from video_processor import VideoProcessor
            from tts_processor import TTSProcessor
            
            output_dir = self._get_output_dir()
            
            # ------------------------------------------------------------
            # 1. Разделение видео и аудио
            # ------------------------------------------------------------
            video_only_path = os.path.join(output_dir, "video_only.mp4")
            audio_path = os.path.join(output_dir, "audio.wav")
            
            if self._file_exists("video_only.mp4") and self._file_exists("audio.wav"):
                self.progress.emit("Видео и аудио уже разделены, пропускаем...")
                video_only = video_only_path
                audio = audio_path
            else:
                self.progress.emit("Разделение видео и аудио...")
                video_proc = VideoProcessor(self.video_path)
                video_only_tmp, audio_tmp = video_proc.split_video_audio()
                video_only = self._move_to_output(video_only_tmp, "video_only.mp4")
                audio = self._move_to_output(audio_tmp, "audio.wav")
            
            if not self._is_running:
                return
            
            # ------------------------------------------------------------
            # 2. Разделение голоса и фона (используем переданный audio_processor)
            # ------------------------------------------------------------
            vocals_path = os.path.join(output_dir, "vocals.wav")
            background_path = os.path.join(output_dir, "background.wav")
            
            if self._file_exists("vocals.wav") and self._file_exists("background.wav"):
                self.progress.emit("Голос и фон уже разделены, пропускаем...")
                vocals = vocals_path
                background = background_path
            else:
                self.progress.emit("Отделение голоса от фоновой музыки...")
                vocals_tmp, background_tmp = self.audio_processor.separate_voice_background(audio)
                vocals = self._move_to_output(vocals_tmp, "vocals.wav")
                background = self._move_to_output(background_tmp, "background.wav")
            
            if not self._is_running:
                return
            
            # ------------------------------------------------------------
            # 3. Транскрибация (голос -> текст)
            # ------------------------------------------------------------
            transcription_file = "transcription.txt"
            if self._file_exists(transcription_file):
                self.progress.emit("Транскрипция уже есть, загружаем...")
                text = self._read_text_file(transcription_file)
            else:
                self.progress.emit("Транскрибация аудио...")
                text = self.audio_processor.transcribe_audio(vocals)
                self._write_text_file(transcription_file, text)
            
            if not self._is_running:
                return
            
            # ------------------------------------------------------------
            # 4. Перевод текста
            # ------------------------------------------------------------
            translation_file = "translated.txt"
            if self._file_exists(translation_file):
                self.progress.emit("Перевод уже есть, загружаем...")
                translated_text = self._read_text_file(translation_file)
            else:
                self.progress.emit("Перевод текста...")
                translated_text = self.audio_processor.translate_text(text)
                self._write_text_file(translation_file, translated_text)
            
            if not self._is_running:
                return
            
            # ------------------------------------------------------------
            # 5. Клонирование голоса и генерация речи
            # ------------------------------------------------------------
            tts_output = os.path.join(output_dir, "generated_speech.wav")
            model_dir = os.path.join(output_dir, "voice_model")
            
            if self._file_exists("generated_speech.wav"):
                self.progress.emit("Сгенерированная речь уже есть, пропускаем...")
            else:
                self.progress.emit("Клонирование голоса и генерация речи...")
                tts_proc = TTSProcessor()
                tts_proc.clone_voice(vocals, model_dir)
                tts_proc.generate_speech(translated_text, tts_output)
            
            if not self._is_running:
                return
            
            # ------------------------------------------------------------
            # 6. Финальная сборка видео
            # ------------------------------------------------------------
            final_output = os.path.join(output_dir, "final_output.mp4")
            if self._file_exists("final_output.mp4"):
                self.progress.emit("Финальное видео уже существует, пропускаем сборку.")
            else:
                self.progress.emit("Сборка финального видео...")
                video_proc = VideoProcessor(self.video_path)
                video_proc.combine_video_audio_with_background(
                    video_only, tts_output, background, final_output
                )
            
            if self._is_running:
                self.progress.emit(f"Готово! Результат в папке {output_dir}")
                self.finished.emit()
            
        except Exception as e:
            if self._is_running:
                error_trace = traceback.format_exc()
                self._log_error_to_file(str(e), error_trace)
                self.error.emit(f"{str(e)}\n\nПодробности записаны в лог: {self.log_file_path}")
    
    # --- вспомогательные методы для работы с файлами в output ---
    def _get_output_dir(self):
        base_dir = os.path.dirname(self.video_path)
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _file_exists(self, filename):
        return os.path.exists(os.path.join(self._get_output_dir(), filename))
    
    def _read_text_file(self, filename):
        path = os.path.join(self._get_output_dir(), filename)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _write_text_file(self, filename, content):
        path = os.path.join(self._get_output_dir(), filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _move_to_output(self, src_path, dest_filename):
        if not src_path or not os.path.exists(src_path):
            return None
        dest_path = os.path.join(self._get_output_dir(), dest_filename)
        if os.path.abspath(src_path) != os.path.abspath(dest_path):
            import shutil
            shutil.move(src_path, dest_path)
        return dest_path
    
    def stop(self):
        self._is_running = False

# ----------------------------------------------------------------------
# GUI приложение
# ----------------------------------------------------------------------
class VideoDubbingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Дубляж видео")
        self.setGeometry(100, 100, 1000, 800)
        
        # Настройка файлового логирования
        self.log_file = setup_file_logging()
        logging.info(f"Лог-файл: {self.log_file}")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        self.select_video_btn = QPushButton("Выбрать видео")
        self.select_video_btn.clicked.connect(self.select_video)
        layout.addWidget(self.select_video_btn)
        
        self.video_label = QLabel("Видео файл не выбран")
        layout.addWidget(self.video_label)
        
        self.process_btn = QPushButton("Начать обработку")
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)
        
        self.cancel_btn = QPushButton("Отменить")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        layout.addWidget(self.cancel_btn)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Статус: Ожидание")
        layout.addWidget(self.status_label)
        
        self.log_area = QTextEdit()
        self.log_area.setMinimumHeight(400)
        layout.addWidget(self.log_area)
        
        # Логирование в текстовое поле GUI
        log_handler = QTextEditLogger(self.log_area)
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_handler)
        logging.getLogger().setLevel(logging.INFO)
        
        self.video_path = None
        self.processing_thread = None
        self.audio_processor = None   # будет создан в основном потоке после выбора видео
        
        signal.signal(signal.SIGINT, self.handle_sigint)
    
    def handle_sigint(self, signum, frame):
        self.cancel_processing()
    
    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео файл", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if file_name:
            self.video_path = file_name
            self.video_label.setText(f"Выбран видео файл: {file_name}")
            
            # Инициализируем AudioProcessor в основном потоке (здесь загрузятся все модели)
            self.status_label.setText("Статус: Загрузка моделей... (это может занять несколько минут)")
            QApplication.processEvents()  # обновляем GUI
            
            try:
                from audio_processor import AudioProcessor
                self.audio_processor = AudioProcessor()
                self.status_label.setText("Статус: Модели загружены")
                self.check_ready()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модели:\n{str(e)}")
                logging.error(f"Ошибка загрузки моделей: {e}", exc_info=True)
                self.audio_processor = None
                self.status_label.setText("Статус: Ошибка загрузки моделей")
    
    def check_ready(self):
        self.process_btn.setEnabled(self.video_path is not None and self.audio_processor is not None)
    
    def process_video(self):
        self.process_btn.setEnabled(False)
        self.select_video_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        self.processing_thread = ProcessingThread(self.video_path, self.log_file, self.audio_processor)
        self.processing_thread.progress.connect(self.update_status)
        self.processing_thread.error.connect(self.handle_error)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()
    
    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Подтверждение', 'Вы уверены, что хотите отменить обработку?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.processing_thread.stop()
                self.processing_thread.wait()
                self.status_label.setText("Статус: Отменено пользователем")
                self.processing_finished()
    
    def update_status(self, message):
        self.status_label.setText(f"Статус: {message}")
        logging.info(message)
    
    def handle_error(self, error_message):
        self.status_label.setText(f"Ошибка: {error_message}")
        logging.error(error_message)
        self.processing_finished()
    
    def processing_finished(self):
        self.process_btn.setEnabled(True)
        self.select_video_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

def main():
    app = QApplication(sys.argv)
    window = VideoDubbingApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
