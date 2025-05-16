import os
import time
import torch
import shutil
import librosa
import multiprocessing as mp
from glob import glob
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

# ========== Utilities ==========

def print_colored(msg, color="green"):
    colors = {
        "green": "\033[92m", "cyan": "\033[96m",
        "yellow": "\033[93m", "red": "\033[91m",
        "magenta": "\033[95m", "bold": "\033[1m", "reset": "\033[0m"
    }
    timestamp = time.strftime("%H:%M:%S")
    print(f"{colors[color]}[{timestamp}] {msg}{colors['reset']}")

def log_gpu_memory(device):
    device_idx = int(device.split(":")[1])
    allocated = torch.cuda.memory_allocated(device_idx) / 1e9
    reserved = torch.cuda.memory_reserved(device_idx) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(device_idx) / 1e9
    print_colored(
        f"GPU {device} Memory: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB, Max Allocated={max_allocated:.2f} GB",
        "yellow"
    )

def seconds_to_srt_time(seconds):
    if seconds is None:
        return None
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    ms = int((s % 1) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"

# ========== Worker ==========

def transcribe_worker(device, task_queue, output_dir, processed_dir):
    torch.cuda.set_device(int(device.split(":")[1]))
    print_colored(f"🔥 Warming up {device}...", "yellow")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "./whisper_model",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("./whisper_model")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=device,
        return_timestamps="word",
    )
    # warm-up
    silence = torch.zeros(16000)
    pipe({"array": silence.numpy(), "sampling_rate": 16000})
    torch.cuda.empty_cache()
    print_colored(f"✅ {device} ready for transcription", "green")

    while not task_queue.empty():
        try:
            audio_file = task_queue.get_nowait()
        except:
            break

        start_time = time.time()
        print_colored(f"🎧 Starting: {os.path.basename(audio_file)} on {device}", "yellow")
        log_gpu_memory(device)

        try:
            audio, sr = librosa.load(audio_file, sr=16000)
            result = pipe({"array": audio, "sampling_rate": sr}, chunk_length_s=30)
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            srt_path = os.path.join(output_dir, f"{base_name}.srt")

            with open(srt_path, "w", encoding="utf-8") as f:
                chunks = result.get("chunks", result.get("segments", []))
                for idx, chunk in enumerate(chunks, 1):
                    start = chunk.get("timestamp", [None, None])[0]
                    end = chunk.get("timestamp", [None, None])[1]
                    text = chunk.get("text", "").strip()
                    if not text or start is None or end is None:
                        continue
                    f.write(f"{idx}\n")
                    f.write(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n")
                    f.write(f"{text}\n\n")

            shutil.move(audio_file, os.path.join(processed_dir, os.path.basename(audio_file)))
            torch.cuda.empty_cache()

            duration = time.time() - start_time
            print_colored(f"✅ Finished: {base_name}.srt on {device} in {duration:.2f} seconds", "cyan")
            log_gpu_memory(device)
        except Exception as e:
            print_colored(f"Error processing {os.path.basename(audio_file)} on {device}: {e}", "red")

# ========== Main ==========

def main():
    script_start = time.time()
    audio_dir = "./unprocessed"
    output_dir = "./output"
    processed_dir = "./processed"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    audio_files = sorted(
        glob(os.path.join(audio_dir, "*.[mM][pP]3")) +
        glob(os.path.join(audio_dir, "*.[wW][aA][vV]")) +
        glob(os.path.join(audio_dir, "*.[fF][lL][aA][cC]")) +
        glob(os.path.join(audio_dir, "*.[oO][gG][gG]")) +
        glob(os.path.join(audio_dir, "*.[mM]4[aA]"))
    )
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print_colored(f"Devices found: {devices}")
    print_colored(f"Total audio files to process: {len(audio_files)}", "yellow")

    if not audio_files:
        print_colored("No audio files found in ./unprocessed")
        return

    task_queue = mp.Queue()
    for f in audio_files:
        task_queue.put(f)

    print_colored("🚀 Launching transcription workers...", "magenta")
    processes = []
    for device in devices:
        p = mp.Process(target=transcribe_worker, args=(device, task_queue, output_dir, processed_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_runtime = time.time() - script_start
    print_colored(f"✨ All done! Total runtime: {total_runtime:.2f} seconds", "green")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
