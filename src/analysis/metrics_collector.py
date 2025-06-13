#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import glob
import pandas as pd
import re

def extract_model_info(model_string):
    """Extract model family and size from model string."""
    if "distil-whisper" in model_string:
        family = "whisper"
        match = re.search(r'distil-(small|medium|large|tiny)', model_string)
        size = match.group(1) if match else "unknown"
    elif "wav2vec2" in model_string:
        family = "wav2vec2"
        size = "large" if "large" in model_string else "base"
    elif "hubert" in model_string:
        family = "hubert"
        size = "large" if "large" in model_string else "base"
    elif "wavlm" in model_string:
        family = "wavlm"
        size = "large" if "large" in model_string else "base"
    elif "unispeech" in model_string:
        family = "unispeech"
        size = "large" if "large" in model_string else "base"
    elif "speecht5" in model_string:
        family = "speecht5"
        size = "base"
    else:
        family = "unknown"
        size = "unknown"
    
    return family, size

def scan_for_json_metrics(directory="."):
    # Use recursive glob to find files in subdirectories
    json_files = glob.glob(os.path.join(directory, "**", "power_metrics_*.json"), recursive=True)
    
    print(f"Found {len(json_files)} power metrics JSON files")
    
    all_metrics = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                metrics = json.load(f)
                
                filename = os.path.basename(json_file)
                match = re.search(r'power_metrics_(.+)_(.+)_\d{4}-\d{2}-\d{2}', filename)
                if match:
                    model_name = match.group(1)
                    precision = match.group(2)
                    model_family, model_size = extract_model_info(model_name)
                    
                    # Add model information to metrics
                    metrics["model_name"] = model_name
                    metrics["precision"] = precision
                    metrics["model_family"] = model_family
                    metrics["model_size"] = model_size
                    
                    all_metrics.append(metrics)
                    print(f"Processed: {json_file}")
                else:
                    print(f"Couldn't extract model info from filename: {filename}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return all_metrics

def parse_log_file(log_file):
    metrics = []
    
    with open(log_file, 'r') as f:
        content = f.read()
        model_entries = re.split(r'(?=Model:)', content)
        
        for entry in model_entries:
            entry = entry.strip()
            if not entry:
                continue
                
            current_metric = {}
            
            model_match = re.search(r'Model: ([^,]+), Precision: ([^,]+)', entry)
            if model_match:
                current_metric['model'] = model_match.group(1).strip()
                current_metric['precision'] = model_match.group(2).strip()
                
                rtf_match = re.search(r'RTF: ([0-9.]+)', entry)
                if rtf_match:
                    current_metric['rtf'] = float(rtf_match.group(1))
                
                wer_match = re.search(r'WER: ([0-9.]+)', entry)
                if wer_match:
                    current_metric['wer'] = float(wer_match.group(1))
                
                energy_match = re.search(r'Total Energy Used \(Joules\): ([0-9.]+)', entry)
                if energy_match:
                    current_metric['total_energy_joule'] = float(energy_match.group(1))
                
                epas_match = re.search(r'EPAS: ([0-9.]+)', entry)
                if epas_match:
                    current_metric['epas'] = float(epas_match.group(1))
                
                mem_mean_match = re.search(r'Mean Mem MB: ([0-9.]+)', entry)
                if mem_mean_match:
                    current_metric['mean_memory_mb'] = float(mem_mean_match.group(1))
                
                mem_max_match = re.search(r'Max Mem MB: ([0-9.]+)', entry)
                if mem_max_match:
                    current_metric['max_memory_mb'] = float(mem_max_match.group(1))
                
                gme_match = re.search(r'GME: ([0-9.]+)', entry)
                if gme_match:
                    current_metric['gme'] = float(gme_match.group(1))
                
                hur_match = re.search(r'HUR: ([0-9.]+)', entry)
                if hur_match:
                    current_metric['hur'] = float(hur_match.group(1))
                
                exec_time_match = re.search(r'Execution Time: ([0-9]+)m([0-9.]+)s', entry)
                if exec_time_match:
                    mins = int(exec_time_match.group(1))
                    secs = float(exec_time_match.group(2))
                    current_metric['execution_time'] = f"{mins}m{secs}s"
                    current_metric['processing_time_seconds'] = mins * 60 + secs
                
                if 'model' in current_metric:
                    model_path = current_metric['model']
                    current_metric['model_name'] = model_path.split('/')[-1]
                    model_family, model_size = extract_model_info(model_path)
                    current_metric['model_family'] = model_family
                    current_metric['model_size'] = model_size
                
                if 'model' in current_metric and 'precision' in current_metric:
                    metrics.append(current_metric)
    
    print(f"Extracted {len(metrics)} model entries from log file")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Collect and consolidate metrics")
    parser.add_argument("--log_file", help="Log file containing experiment results")
    parser.add_argument("--metrics_dir", default=".", help="Directory with JSON metrics files")
    parser.add_argument("--output_csv", default="asr_metrics_results.csv", help="Output CSV file for consolidated metrics")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    all_metrics = []
    
    print(f"Scanning for JSON metrics in: {args.metrics_dir}")
    json_metrics = scan_for_json_metrics(args.metrics_dir)
    if json_metrics:
        print(f"Found {len(json_metrics)} entries from JSON files")
        all_metrics.extend(json_metrics)
        
        if args.verbose:
            for idx, metric in enumerate(json_metrics):
                print(f"JSON Metric {idx+1}:")
                for key, value in sorted(metric.items()):
                    print(f"  {key}: {value}")
    else:
        print("No JSON metrics found")
    
    if args.log_file and os.path.exists(args.log_file):
        print(f"Parsing log file: {args.log_file}")
        log_metrics = parse_log_file(args.log_file)
        
        if args.verbose:
            for idx, metric in enumerate(log_metrics):
                print(f"Log Metric {idx+1}:")
                for key, value in sorted(metric.items()):
                    print(f"  {key}: {value}")
        
        if not all_metrics:
            all_metrics = log_metrics
        else:
            print("Attempting to merge JSON and log metrics")
            
            json_lookup = {}
            for metric in json_metrics:
                if "model_name" in metric and "precision" in metric:
                    key = (metric["model_name"], metric["precision"])
                    json_lookup[key] = metric
            
            merged_metrics = []
            
            for log_metric in log_metrics:
                model_name = log_metric.get("model_name", "")
                precision = log_metric.get("precision", "")
                
                key = (model_name, precision)
                if key in json_lookup:
                    merged = {**json_lookup[key], **log_metric}
                    merged_metrics.append(merged)
                    print(f"Merged metrics for {model_name} ({precision})")
                else:
                    merged_metrics.append(log_metric)
                    print(f"Added log metrics for {model_name} ({precision})")
            
            for key, json_metric in json_lookup.items():
                model_name, precision = key
                if not any(m.get("model_name") == model_name and m.get("precision") == precision for m in merged_metrics):
                    merged_metrics.append(json_metric)
                    print(f"Added JSON-only metrics for {model_name} ({precision})")
            
            all_metrics = merged_metrics
    
    if all_metrics:
        normalized_metrics = []
        for metric in all_metrics:
            normalized = {}
            for key, value in metric.items():
                norm_key = key.strip().lower().replace(" ", "_")
                normalized[norm_key] = value
            normalized_metrics.append(normalized)
        
        df = pd.DataFrame(normalized_metrics)
        
        if "total_audio_duration" not in df.columns:
            # estimate from RTF and processing time if available
            if "processing_time_seconds" in df.columns and "rtf" in df.columns:
                # audio_duration = processing_time / rtf
                df["total_audio_duration"] = df["processing_time_seconds"] / df["rtf"]
                print("Calculated total_audio_duration from processing time and RTF")
            else:
                # Use a default value (e.g., the test set duration)
                df["total_audio_duration"] = 0
                print("Using default audio duration")
        
        if "total_energy_joule" in df.columns and "total_audio_duration" in df.columns and "epas" not in df.columns:
            df["epas"] = df["total_energy_joule"] / df["total_audio_duration"]
            print("Calculated EPAS from total energy and audio duration")
        
        if "processing_time_seconds" in df.columns and "total_audio_duration" in df.columns and "rtf" not in df.columns:
            df["rtf"] = df["processing_time_seconds"] / df["total_audio_duration"]
            print("Calculated RTF from processing time and audio duration")
        
        print("\nMetrics Summary:")
        for idx, row in df.iterrows():
            model = row.get("model_name", "Unknown")
            precision = row.get("precision", "Unknown")
            rtf = row.get("rtf", "N/A")
            epas = row.get("epas", "N/A")
            wer = row.get("wer", "N/A")
            
            print(f"  {model} ({precision}): RTF={rtf}, WER={wer}, EPAS={epas}")
        
        df.to_csv(args.output_csv, index=False)
        print(f"\nConsolidated metrics saved to {args.output_csv}")
        print(f"File contains data for {len(df)} model configurations")
    else:
        print("No metrics found to consolidate")

if __name__ == "__main__":
    main()
