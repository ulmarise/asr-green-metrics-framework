#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# define weighting schemes for different contexts
WEIGHTING_SCHEMES = {
    "balanced": {
        "name": "Balanced Score",
        "description": "General purpose weighting that balances all aspects",
        "weights": {
            "wer": 0.25,     
            "rtf": 0.20,      
            "epas": 0.20,     
            "latency_p95_ms": 0.15,  
            "gme": 0.10,     
            "hur": 0.10       
        }
    },
    "mobile": {
        "name": "Mobile Edge Focus",  
        "description": "Optimized for battery-powered and resource-constrained devices",
        "weights": {
            "epas": 0.30,    
            "gme": 0.25,      
            "rtf": 0.20,      
            "wer": 0.15,      
            "latency_p95_ms": 0.05,  
            "hur": 0.05      
        }
    },
    "realtime": {
        "name": "Real-Time Applications",
        "description": "Optimized for voice assistants and interactive systems",
        "weights": {
            "latency_p95_ms": 0.30,  
            "rtf": 0.25,      
            "wer": 0.25,     
            "epas": 0.10,     
            "gme": 0.05,      
            "hur": 0.05      
        }
    },
    "server": {
        "name": "Server-Side Batch Processing",
        "description": "Optimized for large-scale cloud deployments",
        "weights": {
            "epas": 0.35,     
            "wer": 0.25,      
            "gme": 0.20,      
            "rtf": 0.10,     
            "hur": 0.10,      
            "latency_p95_ms": 0.00  
        }
    },
    "custom": {
        "name": "Custom Weights",
        "description": "User-defined weighting",
        "weights": {}  # Will be populated from command line or config file
    }
}

# fefine which metrics use higher=better scale
HIGHER_IS_BETTER = {
    "wer": False,            
    "rtf": False,            
    "epas": False,            
    "latency_p95_ms": False,  
    "gme": True,              
    "hur": False              
}

def normalize_metric(values, higher_is_better=False):
    values = np.array(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    if min_val == max_val:
        return np.ones_like(values)
    
    normalized = (values - min_val) / (max_val - min_val)
    
    # invert if lower values are better
    if not higher_is_better:
        normalized = 1 - normalized
        
    return normalized

def normalize_hur(values):
    values = np.array(values)
    normalized = np.zeros_like(values, dtype=float)
    
    # for values between 60-90% (optimal range)
    mask_optimal = (60 <= values) & (values <= 90)
    normalized[mask_optimal] = 0.8 + (values[mask_optimal] - 60) * 0.2 / 30
    
    # for values below 60% (under-utilization)
    mask_under = values < 60
    normalized[mask_under] = values[mask_under] * 0.8 / 60
    
    # for values above 90% (over-utilization)
    mask_over = values > 90
    normalized[mask_over] = 1.0 - (values[mask_over] - 90) * 0.5 / 10
    
    return normalized
    
def calculate_weighted_scores(df, weighting_scheme):
    weights = weighting_scheme["weights"]
    result_df = df.copy()
    
    result_df["green_score"] = 0.0
    result_df["efficiency_score"] = 0.0 
    result_df["performance_score"] = 0.0  
    result_df["accuracy_score"] = 0.0    
    
    for metric, weight in weights.items():
        if metric not in df.columns:
            continue
        
        if weight == 0:
            continue
            
        if metric == "hur":
            normalized = normalize_hur(df[metric])
        else:
            normalized = normalize_metric(df[metric], HIGHER_IS_BETTER.get(metric, False))
        
        norm_col = f"{metric}_norm"
        result_df[norm_col] = normalized
        
        result_df["green_score"] += normalized * weight
        
        if metric in ["epas", "gme"]:
            result_df["efficiency_score"] += normalized * (weight / sum(weights.get(m, 0) for m in ["epas", "gme"]))
        elif metric in ["rtf", "latency_p95_ms", "hur"]:
            total_perf_weight = sum(weights.get(m, 0) for m in ["rtf", "latency_p95_ms", "hur"])
            if total_perf_weight > 0:
                result_df["performance_score"] += normalized * (weight / total_perf_weight)
        elif metric == "wer":
            result_df["accuracy_score"] += normalized
    
    if "accuracy_score" not in result_df.columns or result_df["accuracy_score"].isna().all():
        result_df["accuracy_score"] = 0.5  
    
    if "efficiency_score" not in result_df.columns or result_df["efficiency_score"].isna().all():
        result_df["efficiency_score"] = 0.5
        
    if "performance_score" not in result_df.columns or result_df["performance_score"].isna().all():
        result_df["performance_score"] = 0.5
    
    return result_df

def plot_green_scores(df, output_dir, scheme_name):
    """Create bar chart of green scores."""
    plt.figure(figsize=(12, 6))
    
    if "model_display" not in df.columns:
        if "model" in df.columns:
            df["model_display"] = df["model"].apply(lambda x: x.split("/")[-1] if isinstance(x, str) and "/" in x else x)
        elif "model_name" in df.columns:
            df["model_display"] = df["model_name"]
        else:
            df["model_display"] = "Unknown"
        
        df["model_display"] = df["model_display"] + " (" + df["precision"] + ")"
    
    df_sorted = df.sort_values(by="green_score", ascending=False)
    
    sns.barplot(x="model_display", y="green_score", hue="precision", data=df_sorted)
    plt.title(f"Green Score - {scheme_name}", fontsize=14)
    plt.ylabel("Green Score (higher is better)")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    safe_scheme_name = scheme_name.lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_scheme_name = ''.join(c for c in safe_scheme_name if c.isalnum() or c in '_-')
    
    output_path = os.path.join(output_dir, f"green_score_{safe_scheme_name}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Calculate weighted green metrics for ASR models")
    parser.add_argument("metrics_csv", help="CSV file with consolidated metrics")
    parser.add_argument("--output_dir", default="./weighted_metrics", help="Directory to save results")
    parser.add_argument("--html_report", default="weighted_metrics_report.html", help="HTML report output file")
    parser.add_argument("--schemes", default="balanced,mobile,realtime,server", 
                      help="Comma-separated list of weighting schemes to use")
    parser.add_argument("--custom_weights", default=None,
                      help="Custom weights in format 'wer:0.3,rtf:0.2,epas:0.2,latency_p95_ms:0.1,gme:0.1,hur:0.1'")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    try:
        df = pd.read_csv(args.metrics_csv)
        print(f"Loaded data for {len(df)} model configurations")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    if args.custom_weights:
        try:
            custom_weights = {}
            pairs = args.custom_weights.split(",")
            for pair in pairs:
                metric, weight = pair.split(":")
                custom_weights[metric.strip()] = float(weight.strip())
            
            weight_sum = sum(custom_weights.values())
            if weight_sum > 0:
                custom_weights = {k: v/weight_sum for k, v in custom_weights.items()}
            
            WEIGHTING_SCHEMES["custom"]["weights"] = custom_weights
            print(f"Using custom weights: {custom_weights}")
        except Exception as e:
            print(f"Error parsing custom weights: {e}")
    
    if "model_display" not in df.columns:
        if "model" in df.columns:
            df["model_display"] = df["model"].apply(lambda x: x.split("/")[-1] if isinstance(x, str) and "/" in x else x)
        elif "model_name" in df.columns:
            df["model_display"] = df["model_name"]
        else:
            df["model_display"] = "Unknown"
        
        if "precision" in df.columns:
            df["model_display"] = df["model_display"] + " (" + df["precision"] + ")"
    
    schemes = args.schemes.split(",")
    
    results = {}
    
    for scheme in schemes:
        if scheme not in WEIGHTING_SCHEMES:
            print(f"Warning: Unknown weighting scheme '{scheme}', skipping")
            continue
        
        scheme_info = WEIGHTING_SCHEMES[scheme]
        print(f"Applying {scheme_info['name']} weighting scheme...")
        
        missing_metrics = [m for m in scheme_info["weights"] if m not in df.columns and scheme_info["weights"][m] > 0]
        if missing_metrics:
            print(f"Warning: Missing metrics for {scheme}: {', '.join(missing_metrics)}")
            print(f"Available metrics: {', '.join(df.columns)}")
            
            adjusted_weights = {k: v for k, v in scheme_info["weights"].items() if k in df.columns}
            weight_sum = sum(adjusted_weights.values())
            if weight_sum > 0:
                adjusted_weights = {k: v/weight_sum for k, v in adjusted_weights.items()}
                scheme_info = scheme_info.copy()
                scheme_info["weights"] = adjusted_weights
                print(f"Adjusted weights: {adjusted_weights}")
        
        weighted_df = calculate_weighted_scores(df, scheme_info)
         
        # save weighted results to CSV
        output_csv = os.path.join(args.output_dir, f"weighted_metrics_{scheme}.csv")
        weighted_df.to_csv(output_csv, index=False)
        print(f"Weighted metrics saved to {output_csv}")
        
        top3 = weighted_df.sort_values(by="green_score", ascending=False).head(3)
        print(f"Top 3 models for {scheme_info['name']}:")
        for idx, row in top3.iterrows():
            print(f"  {row['model_display']}: {row['green_score']:.3f}")

if __name__ == "__main__":
    main()
