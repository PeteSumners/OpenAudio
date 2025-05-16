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
