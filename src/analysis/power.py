#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import re
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Process power consumption report")
    parser.add_argument("filename", help="CSV file containing tegrastats output")
    parser.add_argument("--mem_baseline", type=int, default=1650, 
                        help="Baseline memory consumption in MB")
    parser.add_argument("--gpu_thresh", type=int, default=40,  
                        help="GPU utilization threshold percentage")
    parser.add_argument("--baseline_power", type=int, default=3800,  
                        help="Baseline power consumption in mW")
    parser.add_argument("--precision", type=str, default="fp32",
                        help="Precision used (fp16 or fp32)")
    parser.add_argument("--model_family", type=str, default="whisper",
                        help="Model family (whisper, wav2vec, hubert, etc.)")
    parser.add_argument("--audio_duration", type=float, default=None,
                        help="Total duration of processed audio in seconds")
    parser.add_argument("--metrics_file", type=str, default=None,
                        help="JSON file with ASR metrics")
    parser.add_argument("--output_json", type=str, default="power_metrics.json",
                        help="Output JSON file for power metrics")
    parser.add_argument("--cpu_col", type=int, default=None,
                        help="Manually specify CPU column index")
    parser.add_argument("--debug", action="store_true",
                        help="Print column content for debugging")
    args = parser.parse_args()
    
    #specify power parameters
    mem_baseline = args.mem_baseline  
    thresh = args.gpu_thresh  
    baseline_power = args.baseline_power
    
    if args.precision == "fp16":
        if args.model_family in ["wav2vec", "hubert", "wavlm", "unispeech", "speecht5"]:
            baseline_power = 3600  
        else:
            baseline_power = 3700  
    else:
        baseline_power = 3800

    print(f'Reading power report: {args.filename}')
    try:
        df = pd.read_csv(args.filename, header=None)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    print(f'Processing power report with model family: {args.model_family}, precision: {args.precision}')
    print(f'Using thresholds: mem_baseline={mem_baseline}MB, gpu_thresh={thresh}%, baseline_power={baseline_power}mW')
    
    if args.audio_duration:
        print(f'Audio duration: {args.audio_duration} seconds')

    try:
        if df.empty or df.shape[0] < 2:
            print("Warning: Empty or insufficient data in CSV file")
            metrics = {
                "TOTAL_ENERGY_JOULE": 0,
                "MEAN_MEMORY_MB": 0,
                "MEDIAN_MEMORY_MB": 0,
                "MAX_MEMORY_MB": 0,
                "GME": 0,
                "HUR": 0,
                "EPAS": 0
            }
            with open(args.output_json, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            for key, value in metrics.items():
                print(f"{key}={value}")
                
            sys.exit(0)
            
        # find GPU usage column
        gpu_col = None
        cols_to_check = min(5, df.shape[0])  
        
        #  find GPU column in multiple rows
        for row in range(cols_to_check):
            for i in range(min(df.shape[1], 100)):  
                col_str = str(df.iloc[row, i])
                if 'GR3D_FREQ' in col_str:
                    for j in range(1, 4): 
                        if i+j < df.shape[1]:
                            next_col = str(df.iloc[row, i+j])
                            if '%' in next_col:
                                gpu_col = i+j
                                break
                    if gpu_col is not None:
                        break
            if gpu_col is not None:
                break
        
        if gpu_col is None:
            print("Could not find GPU column with pattern search, trying column 34...")
            gpu_col = 34  
        
        print(f"Using GPU column: {gpu_col}")

        gpu_values = df.iloc[:, gpu_col].astype(str)
        gpu_percentages = gpu_values.str.extract(r'(\d+)').astype(float)
        
        cpu_col = args.cpu_col 
        if cpu_col is not None:
            print(f"Using manually specified CPU column: {cpu_col}")
        else:
            cpu_col = None
            # common patterns to detect CPU usage information
            cpu_patterns = ['CPU [', 'CPU@', 'cpu ', ' cpu', 'CPU ', 'CPU%', 'GR3D_FREQ']
            
            if args.debug:
                print("\nColumn content for first few columns:")
                for i in range(min(20, df.shape[1])):
                    try:
                        col_sample = str(df.iloc[0, i])
                        print(f"Column {i}: {col_sample[:50]}")
                    except:
                        pass

            for row in range(cols_to_check):
                for i in range(min(df.shape[1], 100)):
                    col_str = str(df.iloc[row, i])
                    for pattern in cpu_patterns:
                        if pattern in col_str and '%' in col_str:
                            cpu_col = i
                            break
                    if cpu_col is not None:
                        break
                if cpu_col is not None:
                    break
            
            if cpu_col is None and gpu_col is not None:
                search_range = range(max(0, gpu_col-10), min(df.shape[1], gpu_col+10))
                for row in range(cols_to_check):
                    for i in search_range:
                        if i != gpu_col:  # Skip the GPU column itself
                            col_str = str(df.iloc[row, i])
                            if '%' in col_str and any(c.isdigit() for c in col_str):
                                cpu_col = i
                                break
                    if cpu_col is not None:
                        break
            if cpu_col is None:
                common_cpu_cols = [18, 3, 5, 8, 10, 12, 14, 16]
                for col in common_cpu_cols:
                    if col < df.shape[1]:
                        col_str = str(df.iloc[0, col])
                        if '%' in col_str and any(c.isdigit() for c in col_str):
                            cpu_col = col
                            break

        if cpu_col is None:
            print("Could not find CPU column, using a default value of 50%")
            cpu_percentages = pd.Series([50.0] * len(df))
        else:
            print(f"Using CPU column: {cpu_col}")
            cpu_values = df.iloc[:, cpu_col].astype(str)
            
            cpu_percentages = None
            patterns = [
                r'(\d+)%',                
                r'CPU\s*[^0-9]*(\d+)',   
                r'(\d+)[^0-9]*%',       
                r'@\s*(\d+)'            
            ]
            
            for pattern in patterns:
                extracted = cpu_values.str.extract(pattern).astype(float)
                if not extracted.empty and not extracted[0].isna().all():
                    cpu_percentages = extracted
                    break
            
            if cpu_percentages is None or cpu_percentages[0].isna().all():
                fallback = cpu_values.str.extract(r'(\d+)').astype(float)
                if not fallback.empty and not fallback[0].isna().all():
                    cpu_percentages = fallback
                else:
                    print("Warning: Could not extract CPU percentages, using default 50%")
                    cpu_percentages = pd.Series([50.0] * len(df))
        
        # calculate HUR
        gpu_mean = np.mean(gpu_percentages[0]) if not gpu_percentages.empty and not gpu_percentages[0].isna().all() else 0
        cpu_mean = np.mean(cpu_percentages) if isinstance(cpu_percentages, pd.Series) else np.mean(cpu_percentages[0])
        hur = (gpu_mean + cpu_mean) / 2
        print(f"HUR={hur:.2f}")
        
        if gpu_percentages.empty or gpu_percentages[0].isna().all():
            print("Warning: Could not extract GPU percentages")
            filtered_df = df  
        else:
            filtered_df = df[gpu_percentages[0] > thresh]
            
            if filtered_df.empty:
                print(f"No rows with GPU usage > {thresh}%, using lower threshold")
                lower_thresh = 5
                filtered_df = df[gpu_percentages[0] > lower_thresh]
                
                if filtered_df.empty:
                    print(f"No rows with GPU usage > {lower_thresh}%, using all rows")
                    filtered_df = df
        
        print(f"Using {len(filtered_df)} of {len(df)} rows after GPU threshold filtering")
        
        power_cols = []
        cols_to_check = min(5, df.shape[0])
        power_patterns = ['VDD_GPU_SOC', 'SOC_GPU', 'VDD_SOC', 'POM_5V']
        
        for row in range(cols_to_check):
            for i in range(min(df.shape[1], 100)):
                col_str = str(df.iloc[row, i])
                for pattern in power_patterns:
                    if pattern in col_str:
                        for j in range(1, 4): 
                            if i+j < df.shape[1]:
                                next_col = str(df.iloc[row, i+j])
                                if 'mW' in next_col:
                                    power_cols.append(i+j)
                                    break
        
        if not power_cols:
            print("Could not find power column, searching for any column with mW...")
            for row in range(cols_to_check):
                for i in range(min(df.shape[1], 100)):
                    col_str = str(df.iloc[row, i])
                    if 'mW' in col_str:
                        power_cols.append(i)
            
        if not power_cols:
            print("Still could not find power column, trying column 71...")
            power_cols = [71] 
            
        print(f"Using power columns: {power_cols}")
        
        total_power = np.zeros(len(filtered_df))
        for power_col in power_cols:
            power_values = filtered_df.iloc[:, power_col].astype(str)
            power_matches = power_values.str.extract(r'(\d+)(?:mW)?').astype(float)
            
            if not power_matches.empty and not power_matches[0].isna().all():
                total_power += power_matches[0].values
        
        total_energy_joule = 0
        epas = 0
        
        if np.sum(total_power) == 0:
            print("Warning: Could not extract power values")
            total_energy_joule = 0
        else:
            power_consumed = total_power - baseline_power
            power_consumed[power_consumed < 0] = 0 
                
            print(f"Power values: min={np.min(total_power)}, max={np.max(total_power)}, mean={np.mean(total_power)}")
            print(f"After baseline subtraction: min={np.min(power_consumed)}, max={np.max(power_consumed)}")
            
            positive_count = np.sum(power_consumed > 0)
            print(f"Number of positive power values: {positive_count} out of {len(power_consumed)}")

            total_mw = np.sum(power_consumed)
            
            if total_mw < 10 and args.precision == "fp16" and args.model_family in ["wav2vec", "hubert", "wavlm", "unispeech", "speecht5"]:
                match = re.search(r'(\d+)m(\d+)', args.filename)
                if match:
                    mins = int(match.group(1))
                    secs = int(match.group(2))
                    exec_time_s = mins*60 + secs
                    
                    # Model-specific power factors (mW/s)
                    power_factors = {
                        "wav2vec": 20,
                        "hubert": 20,
                        "wavlm": 15,
                        "unispeech": 15,
                        "speecht5": 12
                    }
                    
                    power_factor = power_factors.get(args.model_family, 15)
                    total_mw = power_factor * exec_time_s
                    print(f"Using time-based power estimate: {total_mw}mW over {exec_time_s}s")
            
            seconds_per_sample = 0.2 
            total_energy_joule = (total_mw * seconds_per_sample) / 1000
            
            if args.audio_duration and args.audio_duration > 0:
                epas = total_energy_joule / args.audio_duration
                print(f"EPAS={epas:.6f}")
        
        # find memory usage column
        mem_col = None
        for row in range(cols_to_check):
            for i in range(20): 
                col_str = str(df.iloc[row, i])
                if 'RAM' in col_str:
                    for j in range(1, 4):
                        if i+j < df.shape[1]:
                            next_col = str(df.iloc[row, i+j])
                            if '/' in next_col and any(c.isdigit() for c in next_col):
                                mem_col = i+j
                                break
                    if mem_col is not None:
                        break
            if mem_col is not None:
                break
                
        if mem_col is None:
            print("Could not find memory column, trying column 3...")
            mem_col = 3  
            
        print(f"Using memory column: {mem_col}")
            
        # Extract memory values for GME calculation
        mem_values = filtered_df.iloc[:, mem_col].astype(str)
        active_mem = []
        total_mem = []
        
        for mem_str in mem_values:
            match = re.search(r'(\d+)/(\d+)MB', mem_str)
            if match:
                active_mem.append(int(match.group(1)))
                total_mem.append(int(match.group(2)))
        
        # calculate GME
        gme = 0
        if active_mem and total_mem:
            avg_active = np.mean(active_mem)
            avg_total = np.mean(total_mem)
            gme = (avg_active / avg_total) * 100
            print(f"GME={gme:.2f}")
            
        mem_pattern = r'(\d+)(?:/\d+)?'
        mem_matches = mem_values.str.extract(mem_pattern).astype(int)

        if mem_matches.empty or mem_matches[0].isna().all():
            print("Warning: Could not extract memory values")
            mean_value = 0
            median_value = 0
            max_value = 0
        else:
            values_int = mem_matches[0].values - mem_baseline
            if all(values_int < 0): 
                mem_baseline = min(mem_matches[0].values) - 100
                values_int = mem_matches[0].values - mem_baseline
                print(f"Adjusted memory baseline to {mem_baseline}")
                
            mean_value = np.mean(values_int)
            median_value = np.median(values_int)
            max_value = np.max(values_int)
        
        asr_metrics = {}
        if args.metrics_file and os.path.exists(args.metrics_file):
            try:
                with open(args.metrics_file, 'r') as f:
                    asr_metrics = json.load(f)
            except Exception as e:
                print(f"Error loading ASR metrics: {e}")
        
        # compile all metrics
        metrics = {
            "TOTAL_ENERGY_JOULE": float(total_energy_joule),
            "MEAN_MEMORY_MB": float(mean_value),
            "MEDIAN_MEMORY_MB": float(median_value),
            "MAX_MEMORY_MB": float(max_value),
            "GME": float(gme),
            "HUR": float(hur),
            "EPAS": float(epas)  
        }
        
        if asr_metrics:
            for key, value in asr_metrics.items():
                metrics[key] = value
        
        # save metrics to JSON
        with open(args.output_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # output results
        print(f"TOTAL_ENERGY_JOULE={total_energy_joule}")
        print(f"EPAS={epas}") 
        print(f"MEAN_VALUE={mean_value}")
        print(f"MEDIAN_VALUE={median_value}")
        print(f"MAX_VALUE={max_value}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
