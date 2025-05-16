import time
# Record the script start time (after importing the time library)
script_start_time = time.time()

import os
import torch
import shutil
import librosa
import multiprocessing as mp
from glob import glob
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from torch.nn.attention import SDPBackend, sdpa_kernel
from queue import Queue
from threading import Lock


# Function to print colorized messages with timestamp
def print_colored(msg, color="green"):
    colors = {
        "green": "\033[92m",
        "cyan": "\033[96m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "magenta": "\033[95m",
        "bold": "\033[1m",
        "reset": "\033[0m"
    }
    timestamp = time.strftime("%H:%M:%S")
    print(f"{colors[color]}[{timestamp}] {msg}{colors['reset']}")

# Function to log GPU memory usage
def log_gpu_memory(device):
    device_idx = int(device.split(":")[1])
    allocated = torch.cuda.memory_allocated(device_idx) / 1e9  # Convert to GB
    reserved = torch.cuda.memory_reserved(device_idx) / 1e9   # Convert to GB
    max_allocated = torch.cuda.max_memory_allocated(device_idx) / 1e9  # Convert to GB
    print_colored(
        f"GPU {device} Memory: Allocated={allocated:.2f} GB, Reserved(Cached)={reserved:.2f} GB, Max Allocated={max_allocated:.2f} GB",
        "yellow"
    )

# Function to load the model for a device (for warmup)
def warmup(device):
    torch.cuda.set_device(int(device.split(":")[1]))
    print_colored(f"Device set to use {device}")

    # Log memory before warmup
    log_gpu_memory(device)

    # Load model and processor from local directory
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "./whisper_model",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("./whisper_model")

    # Initialize pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=device
    )

    # Generate 1 second of silence at 16 kHz
    sample_rate = 16000
    silence = torch.zeros(sample_rate)
    pipe({"array": silence.numpy(), "sampling_rate": sample_rate})

    # Log memory after warmup
    log_gpu_memory(device)

    # Minimal cleanup
    torch.cuda.empty_cache()
    return device

# Function to format seconds to SRT timestamp (HH:MM:SS,MS)
def seconds_to_srt_time(seconds):
    if seconds is None:
        return None
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# Function to transcribe audio file on specific device and output SRT
def transcribe(audio_file, device, output_dir, processed_dir):
    start_time = time.time()
    print_colored(f"🎧 Starting: {os.path.basename(audio_file)} on {device}", "yellow")

    torch.cuda.set_device(int(device.split(":")[1]))

    # Load model and processor from local directory
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "./whisper_model",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    model.generation_config.cache_implementation = "static"
    model.generation_config.max_new_tokens = 256
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    processor = AutoProcessor.from_pretrained("./whisper_model")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=device,
        return_timestamps=True,
    )

    audio, sr = librosa.load(audio_file, sr=16000)
    sample = {"array": audio, "sampling_rate": sr}

    with sdpa_kernel(SDPBackend.MATH):
        result = pipe(sample, generate_kwargs={"language": "en"}, chunk_length_s=30.0)

    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    srt_path = os.path.join(output_dir, f"{base_name}.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        chunks = result.get("chunks", result.get("segments", []))
        for idx, chunk in enumerate(chunks, 1):
            start = chunk.get("timestamp", [0, 0])[0] if "timestamp" in chunk else chunk.get("start", 0)
            end = chunk.get("timestamp", [0, 0])[1] if "timestamp" in chunk else chunk.get("end", 0)
            text = chunk.get("text", "").strip()
            
            # Skip chunks with invalid timestamps or empty text
            if start is None or end is None or not text:
                continue
                
            start_time_srt = seconds_to_srt_time(start)
            end_time_srt = seconds_to_srt_time(end)
            if start_time_srt is None or end_time_srt is None:
                continue
                
            f.write(f"{idx}\n")
            f.write(f"{start_time_srt} --> {end_time_srt}\n")
            f.write(f"{text}\n\n")

    # Move original audio file to processed dir
    os.makedirs(processed_dir, exist_ok=True)
    shutil.move(audio_file, os.path.join(processed_dir, os.path.basename(audio_file)))

    # Minimal cleanup (process termination handles most memory release)
    torch.cuda.empty_cache()

    duration = time.time() - start_time
    print_colored(f"✅ Finished: {base_name}.srt on {device} in {duration:.2f} seconds", "cyan")

# Wrapper function for child process with memory debugging
def transcribe_wrapper(audio_file, device, output_dir, processed_dir):
    try:
    
        # Log memory before transcription
        log_gpu_memory(device)
        
        transcribe(audio_file, device, output_dir, processed_dir)
        
        # Log memory after transcription and cleanup
        log_gpu_memory(device)
    except Exception as e:
        print_colored(f"Error processing {os.path.basename(audio_file)} on {device}: {e}", "red")

# Main function
def main():
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

    # Setup → warmup timer
    pre_warmup_time = time.time()
    setup_duration = pre_warmup_time - script_start_time
    print_colored(f"🛠️  Setup completed in {setup_duration:.2f} seconds", "cyan")

    print_colored("🔥 Warming up models on all GPUs...", "yellow")
    for device in devices:
        warmup(device)

    warmup_end_time = time.time()
    warmup_duration = warmup_end_time - pre_warmup_time
    print_colored(f"✅ Warmup done in {warmup_duration:.2f} seconds", "green")

    if not audio_files:
        print_colored("No audio files found in ./unprocessed")
        total_duration = time.time() - script_start_time
        print_colored(f"✨ Total script runtime: {total_duration:.2f} seconds", "cyan")
        return

    print_colored("🚀 Launching transcription jobs...", "magenta")

    # Create a queue for audio files
    task_queue = Queue()
    for audio_file in audio_files:
        task_queue.put(audio_file)

    # Track active devices and processes
    active_devices = set()
    device_lock = Lock()
    processes = []

    # Function to assign a task to an available device
    def start_new_task():
        with device_lock:
            for device in devices:
                if device not in active_devices:
                    if not task_queue.empty():
                        audio_file = task_queue.get()
                        active_devices.add(device)
                        print_colored(f"Assigning {os.path.basename(audio_file)} to {device}", "magenta")
                        p = mp.Process(
                            target=transcribe_wrapper,
                            args=(audio_file, device, output_dir, processed_dir)
                        )
                        p.start()
                        return p, device
        return None, None

    # Start initial tasks up to the number of devices
    for _ in range(min(len(audio_files), len(devices))):
        p, device = start_new_task()
        if p:
            processes.append((p, device))

    # Monitor and replace completed processes
    while processes:
        for i, (p, device) in enumerate(processes[:]):  # Iterate over a copy
            if not p.is_alive():
                p.join()
                with device_lock:
                    active_devices.remove(device)
                processes.pop(i)
                # Start a new task if there are more files
                p, new_device = start_new_task()
                if p:
                    processes.append((p, new_device))

    total_duration = time.time() - script_start_time
    print_colored(f"✨ All done! Total runtime: {total_duration:.2f} seconds", "green")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()