#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
import librosa
import os
import sys
import argparse
import time
import json
import numpy as np

def calculate_total_audio_duration(directory):
    total_duration = 0
    supported_extensions = [".flac", ".mp3"]
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            audio_path = os.path.join(directory, filename)
            try:
                duration = librosa.get_duration(path=audio_path)
                total_duration += duration
            except Exception as e:
                print(f"Warning: Could not get duration for {filename}: {e}")
                try:
                    audio, sr = librosa.load(audio_path, sr=None)
                    total_duration += len(audio) / sr
                except Exception as e2:
                    print(f"Error loading {filename}: {e2}")
    
    return total_duration

def get_audio_files(directory):
    supported_extensions = [".flac", ".mp3"]
    audio_files = []
    
    for filename in sorted(os.listdir(directory)):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            audio_files.append(filename)
    
    return audio_files

def get_output_filename(input_filename):
    if input_filename.lower().endswith(".flac"):
        return input_filename[:-5] + ".txt"
    elif input_filename.lower().endswith(".mp3"):
        return input_filename[:-4] + ".txt"
    else:
        return os.path.splitext(input_filename)[0] + ".txt"

def main():
    parser = argparse.ArgumentParser(description="Run Wav2Vec2 ASR model on audio files")
    parser.add_argument("model_id", type=str, help="Model ID from HuggingFace")
    parser.add_argument("precision", type=str, choices=["fp32", "fp16"],
                        help="Precision to use (fp32 or fp16)")
    parser.add_argument("input_directory", type=str, help="Directory containing audio files")
    parser.add_argument("output_directory", type=str, help="Directory to save transcriptions")
    parser.add_argument("--metrics_file", type=str, default="metrics.json",
                        help="File to save metrics data")
    parser.add_argument("--verbose", action="store_true", help="Print verbose logs")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate for audio processing")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum number of new tokens to generate")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if args.precision == "fp16" and torch.cuda.is_available() else torch.float32

    print(f"Loading model {args.model_id} with {args.precision} precision...")
    print(f"Device: {device}, Dtype: {torch_dtype}")
    print(f"Target sample rate: {args.sample_rate}Hz")

    audio_files = get_audio_files(args.input_directory)
    if not audio_files:
        print(f"No supported audio files found in {args.input_directory}")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files: {', '.join(set(f.split('.')[-1].upper() for f in audio_files))} formats")

    model_load_start = time.time()
    try:
        processor = Wav2Vec2Processor.from_pretrained(args.model_id)
        model = Wav2Vec2ForCTC.from_pretrained(args.model_id)
        model = model.to(device).to(torch_dtype)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f} seconds")

    # inference pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=torch_dtype,
        device=device,
    )

    os.makedirs(args.output_directory, exist_ok=True)

    total_audio_duration = calculate_total_audio_duration(args.input_directory)
    print(f"TOTAL_AUDIO_DURATION={total_audio_duration:.2f}s")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)

    results = {}
    latencies = []
    total_processing_time = 0

    print(f"Processing audio files from {args.input_directory}...")
    for filename in audio_files:
        print(f"Transcribing {filename}...")
        file_path = os.path.join(args.input_directory, filename)
        
        try:
            audio_input, sr = librosa.load(file_path, sr=args.sample_rate)
            audio_duration = len(audio_input) / args.sample_rate

            start_time = time.time()
            result = pipe(audio_input)
            end_time = time.time()

            processing_time = end_time - start_time
            latency_ms = processing_time * 1000
            rtf_file = processing_time / audio_duration if audio_duration > 0 else 0

            total_processing_time += processing_time
            latencies.append(latency_ms)

            print(f"File: {filename}, Duration: {audio_duration:.2f}s, Latency: {latency_ms:.2f}ms, RTF: {rtf_file:.3f}")

            transcription = result["text"]
            results[filename] = {
                "text": transcription,
                "latency_ms": latency_ms,
                "rtf": rtf_file,
                "duration": audio_duration
            }

            output_filename = get_output_filename(filename)
            output_file_path = os.path.join(args.output_directory, output_filename)
            with open(output_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(transcription)

            if args.verbose:
                print(f"Saved transcription to {output_file_path}: {transcription[:50]}...")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if not latencies:
        print("No files were successfully processed!")
        sys.exit(1)

    # Calculate overall metrics
    rtf_overall = total_processing_time / total_audio_duration if total_audio_duration > 0 else 0
    percentile_95 = np.percentile(latencies, 95) if latencies else 0

    peak_gpu_memory = 0
    gpu_memory_allocated = 0
    if torch.cuda.is_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  

    metrics = {
        "model_id": args.model_id,
        "precision": args.precision,
        "sample_rate": args.sample_rate,
        "max_new_tokens": args.max_new_tokens,
        "rtf": rtf_overall,
        "total_audio_duration": total_audio_duration,
        "total_processing_time": total_processing_time,
        "latency_p95_ms": percentile_95,
        "model_load_time": model_load_time,
        "peak_gpu_memory_mb": peak_gpu_memory,
        "gpu_memory_allocated_mb": gpu_memory_allocated,
        "gpu_memory_increase_mb": peak_gpu_memory - initial_gpu_memory if torch.cuda.is_available() else 0,
        "files_processed": len(latencies),
        "files_found": len(audio_files)
    }

    print(f"Processing complete. Results saved to {args.output_directory}")
    print(f"Files processed: {len(latencies)}/{len(audio_files)}")
    print(f"RTF={rtf_overall:.3f}")
    print(f"LATENCY_P95_MS={percentile_95:.2f}")
    print(f"PEAK_GPU_MEMORY_MB={peak_gpu_memory:.2f}")

    with open(args.metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {args.metrics_file}")

if __name__ == "__main__":
    main()