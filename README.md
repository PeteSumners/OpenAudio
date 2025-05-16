## 🛠️ Requirements & Setup

This project includes:

* A **parallel Whisper transcription pipeline** with GPU acceleration, multiprocessing, and optional word-level timestamps.
* An **SRT semantic search system** using FAISS for fast vector search, capable of playing the original audio of the top result.

---

### 🔧 Software Requirements

Make sure the following are installed:

#### 🔉 Audio & Transcription

* `ffmpeg` — For audio extraction and conversion
* `torchaudio` — For audio loading (backend-agnostic)
* `openai-whisper` or `faster-whisper` — For speech-to-text
* `torch` + `CUDA` (if using GPU acceleration)
* `transformers` — For embeddings (e.g., `T5EncoderModel`)
* `huggingface_hub` — For model download and caching

#### 🧠 Semantic Search

* `faiss-cpu` or `faiss-gpu` — For fast similarity search
* `numpy`, `scipy`, `scikit-learn` — Standard ML stack
* `librosa`, `soundfile`, `pydub` — For loading and playing audio
* `python-multiprocessing` — For parallel processing (built-in)

---

### 📂 Optional (for advanced playback & GUI integration)

* `pyaudio` or `pygame` — For more flexible audio playback (platform-dependent)
* GUI or frontend integration (e.g., `gradio`, `tkinter`) if needed

---

## 💻 Hardware Requirements

* **Minimum CPU**: 4 cores (8+ recommended for multiprocess scaling)
* **RAM**: 8GB minimum, 16GB+ preferred
* **GPU** (optional but recommended): Any CUDA-compatible GPU (NVIDIA, 6GB+ VRAM ideal)
* **Disk space**: Depends on audio size and embedding cache (typically a few GB)

---

## 🖥️ Operating System Compatibility

| OS                              | Status                 | Notes                                                                                                                                                                                                       |
| ------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ubuntu/Linux (Debian-based)** | ✅ Fully supported      | Native support for `ffmpeg`, GPU drivers, multiprocessing, and audio backends. Development assumed on this platform.                                                                                        |
| **macOS**                       | ⚠️ Partially supported | Whisper and SRT indexing will work, but:<br>• No CUDA GPU support (CPU-only transcription)<br>• Audio playback may require adjustments (`librosa` + `soundfile` backends)<br>• FAISS works with `faiss-cpu` |
| **Windows (10/11)**             | ⚠️ Works with tweaks   | You’ll need:<br>• WSL2 or a proper CUDA+Python toolchain<br>• `ffmpeg` in PATH<br>• Multiprocessing setup must handle Windows's `spawn` mode<br>• Playback may fail without PyAudio/sound backend setup     |



## 🧭 Project Workflow Overview

This project pipeline covers audio downloading, transcription (two modes), and semantic search indexing.

---

### 1. `download.py` — 🔽 **Audio Downloader & Preprocessor**

* Downloads or loads audio files.
* Converts and preprocesses audio to standard format (mono, 16kHz WAV).
* Organizes files into `unprocessed/` (raw) and `processed/` (ready for transcription).

---

### 2. Transcription scripts (choose one or both):

* `sentence_transcribe.py` — 🧵 **Sentence-level Transcription**

  * Transcribes audio at the sentence level.
  * Outputs subtitles to `subs/` directory.
  * Uses the `whisper_model/` directory for models.

* `word_transcribe.py` — ⏱️ **Word-level Timestamp Transcription**

  * Provides detailed word-level timestamps.
  * Also outputs subtitles to `subs/`.
  * Uses models from `whisper_model/`.

---

### 3. `srt_to_faiss.py` — 🧠 **Semantic Search Index Builder**

* Parses `.srt` subtitle files from `subs/`.
* Creates vector embeddings and builds a FAISS index.
* Index can be queried for semantic search.

---

## 🧱 Execution Flow

```bash
# Step 1: Download and preprocess audio
python download.py

# Step 2: Transcribe audio (choose one)
python sentence_transcribe.py      # Sentence-level subtitles
# OR
python word_transcribe.py          # Word-level subtitles

# Step 3: Build FAISS index for semantic search
python srt_to_faiss.py
```
