#!/bin/bash
set -e  # Exit on any error

# ASR models to evaluate
declare -a models=(
  "distil-whisper/distil-small.en"
  "distil-whisper/distil-medium.en" 
  "distil-whisper/distil-large-v2"
  "facebook/wav2vec2-large-960h"
  "facebook/hubert-large-ls960-ft"
  "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
  "microsoft/unispeech-sat-base-100h-libri-ft"
  "microsoft/speecht5_asr"
)

# Precision settings
declare -a precisions=("fp32" "fp16")

# Configuration
DATA_DIR="./data"
OUTPUT_BASE_DIR="./output"
METRICS_BASE_DIR="./metrics_data"
REFERENCE_FILE="./data/trans.txt"  # Default reference file
LOG_FILE="process_log.txt"

# Create directories
mkdir -p $METRICS_BASE_DIR
mkdir -p $OUTPUT_BASE_DIR

# Initialize log
> $LOG_FILE
echo "=== ASR Green Metrics Framework Evaluation Started at $(date) ===" | tee -a $LOG_FILE

# Calculate total audio duration
echo "Calculating total audio duration..." | tee -a $LOG_FILE
TOTAL_AUDIO_DURATION=0

# Calculate duration for FLAC files
if command -v soxi &> /dev/null && find $DATA_DIR -name "*.flac" -type f | head -1 | grep -q .; then
    FLAC_DURATION=$(find $DATA_DIR -name "*.flac" -exec soxi -D {} \; 2>/dev/null | awk '{s+=$1} END {print s+0}')
    TOTAL_AUDIO_DURATION=$(echo "$TOTAL_AUDIO_DURATION + $FLAC_DURATION" | bc -l)
    echo "FLAC files duration: $FLAC_DURATION seconds" | tee -a $LOG_FILE
fi

# Calculate duration for MP3 files
if find $DATA_DIR -name "*.mp3" -type f | head -1 | grep -q .; then
    MP3_DURATION=$(python3 -c "
import librosa
import os
import glob
total = 0
for f in glob.glob('$DATA_DIR/**/*.mp3', recursive=True):
    try:
        duration = librosa.get_duration(path=f)
        total += duration
    except Exception as e:
        print(f'Warning: Could not process {f}: {e}')
print(f'{total:.2f}')
" 2>/dev/null || echo "0")
    TOTAL_AUDIO_DURATION=$(echo "$TOTAL_AUDIO_DURATION + $MP3_DURATION" | bc -l)
    echo "MP3 files duration: $MP3_DURATION seconds" | tee -a $LOG_FILE
fi

echo "Total audio duration: $TOTAL_AUDIO_DURATION seconds" | tee -a $LOG_FILE

if [ "$(echo "$TOTAL_AUDIO_DURATION <= 0" | bc -l)" -eq 1 ]; then
    echo "Warning: No audio files found or duration calculation failed. Please check $DATA_DIR" | tee -a $LOG_FILE
    echo "Supported formats: FLAC, MP3" | tee -a $LOG_FILE
    exit 1
fi

# Function to get model family from model name
get_model_family() {
    local model=$1
    if [[ $model == *"distil-whisper"* ]] || [[ $model == *"whisper"* ]]; then
        echo "whisper"
    elif [[ $model == *"wav2vec2"* ]]; then
        echo "wav2vec"
    elif [[ $model == *"hubert"* ]]; then
        echo "hubert"
    elif [[ $model == *"wavlm"* ]]; then
        echo "wavlm"
    elif [[ $model == *"unispeech"* ]]; then
        echo "unispeech"
    elif [[ $model == *"speecht5"* ]]; then
        echo "speecht5"
    else
        echo "unknown"
    fi
}

run_asr_model() {
    local model=$1
    local precision=$2
    local output_dir=$3
    local metrics_json=$4
    local time_file=$5
    
    local model_family=$(get_model_family $model)
    
    echo "Running ASR model: $model (family: $model_family, precision: $precision)" | tee -a $LOG_FILE
    
    case $model_family in
        "whisper")
            { time python src/models/whisper.py "$model" "$precision" "$DATA_DIR" "$output_dir" \
                --metrics_file "$metrics_json" --verbose --recursive; } 2> "$time_file"
            ;;
        "wav2vec")
            { time python src/models/wav2vec2.py "$model" "$precision" "$DATA_DIR" "$output_dir" \
                --metrics_file "$metrics_json" --verbose; } 2> "$time_file"
            ;;
        "hubert")
            { time python src/models/hubert.py "$model" "$precision" "$DATA_DIR" "$output_dir" \
                --metrics_file "$metrics_json" --verbose; } 2> "$time_file"
            ;;
        "wavlm")
            { time python src/models/wavlm.py "$model" "$precision" "$DATA_DIR" "$output_dir" \
                --metrics_file "$metrics_json" --verbose; } 2> "$time_file"
            ;;
        "unispeech")
            { time python src/models/unispeech.py "$model" "$precision" "$DATA_DIR" "$output_dir" \
                --metrics_file "$metrics_json" --verbose; } 2> "$time_file"
            ;;
        "speecht5")
            { time python src/models/speecht5.py "$model" "$precision" "$DATA_DIR" "$output_dir" \
                --metrics_file "$metrics_json" --verbose; } 2> "$time_file"
            ;;
        *)
            echo "Error: Unknown model family for $model" | tee -a $LOG_FILE
            return 1
            ;;
    esac
}

for model in "${models[@]}"; do
    for precision in "${precisions[@]}"; do
        echo "========================================" | tee -a $LOG_FILE
        echo "Starting experiment: $model with $precision precision" | tee -a $LOG_FILE
        echo "Time: $(date)" | tee -a $LOG_FILE
        
        # Extract model name for file naming
        model_name=$(echo $model | sed 's/.*\///' | tr '/' '_')
        CURRENT_DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
        
        # Set up directories
        METRICS_DIR="$METRICS_BASE_DIR/${model_name}_${precision}"
        OUTPUT_DIR="$OUTPUT_BASE_DIR/${model_name}_${precision}"
        mkdir -p "$METRICS_DIR" "$OUTPUT_DIR"
        
        # File paths
        METRICS_JSON="$METRICS_DIR/asr_metrics_${model_name}_${precision}.json"
        TIME_FILE="$METRICS_DIR/time.txt"
        TEGRASTATS_LOGFILE="$METRICS_DIR/teg_${model_name}_${precision}_$CURRENT_DATETIME.csv"
        
        # Start power monitoring with tegrastats
        echo "Starting power monitoring (tegrastats with 200ms interval)..." | tee -a $LOG_FILE
        if command -v tegrastats &> /dev/null; then
            sudo tegrastats --interval 200 --logfile "$TEGRASTATS_LOGFILE" &
            TEGRASTATS_PID=$!
            sleep 2  # Give tegrastats time to initialize
            echo "Tegrastats started with PID: $TEGRASTATS_PID" | tee -a $LOG_FILE
        else
            echo "Warning: tegrastats not available. Power monitoring disabled." | tee -a $LOG_FILE
            TEGRASTATS_PID=""
        fi
        
        # Run ASR model evaluation
        echo "Starting ASR transcription..." | tee -a $LOG_FILE
        if run_asr_model "$model" "$precision" "$OUTPUT_DIR" "$METRICS_JSON" "$TIME_FILE"; then
            echo "ASR transcription completed successfully" | tee -a $LOG_FILE
        else
            echo "Error: ASR transcription failed for $model with $precision" | tee -a $LOG_FILE
            if [ -n "$TEGRASTATS_PID" ]; then
                sudo kill $TEGRASTATS_PID 2>/dev/null || true
            fi
            continue
        fi
        
        # Stop power monitoring
        if [ -n "$TEGRASTATS_PID" ]; then
            echo "Stopping power monitoring..." | tee -a $LOG_FILE
            sudo kill $TEGRASTATS_PID 2>/dev/null || true
            sleep 1
        fi
        
        # Process execution time
        EXEC_TIME="0m0s"
        if [ -f "$TIME_FILE" ]; then
            EXEC_TIME=$(grep "real" "$TIME_FILE" | awk '{print $2}' || echo "0m0s")
        fi
        echo "Execution time: $EXEC_TIME" | tee -a $LOG_FILE
        
        # Convert execution time to seconds
        EXEC_TIME_SEC=$(echo "$EXEC_TIME" | awk -F 'm|s' '{print $1*60+$2}' || echo "0")
        
        # Process tegrastats output if available
        if [ -f "$TEGRASTATS_LOGFILE" ] && [ -s "$TEGRASTATS_LOGFILE" ]; then
            OUTPUT_FILE="$METRICS_DIR/tegrastats_processed_${model_name}_${precision}_$CURRENT_DATETIME.csv"
            cat "$TEGRASTATS_LOGFILE" | tr -s ' ' ',' > "$OUTPUT_FILE"
            echo "Tegrastats output processed: $OUTPUT_FILE" | tee -a $LOG_FILE
        else
            echo "Warning: No tegrastats output available" | tee -a $LOG_FILE
            OUTPUT_FILE=""
        fi
        
        # Get audio duration from ASR metrics or use total
        AUDIO_DURATION=""
        if [ -f "$METRICS_JSON" ]; then
            AUDIO_DURATION=$(python3 -c "
import json
try:
    with open('$METRICS_JSON', 'r') as f:
        data = json.load(f)
    print(data.get('total_audio_duration', ''))
except:
    print('')
" 2>/dev/null)
        fi
        
        if [ -z "$AUDIO_DURATION" ] || [ "$(echo "$AUDIO_DURATION <= 0" | bc -l 2>/dev/null || echo 1)" -eq 1 ]; then
            AUDIO_DURATION=$TOTAL_AUDIO_DURATION
            echo "Using pre-calculated total audio duration: $AUDIO_DURATION seconds" | tee -a $LOG_FILE
        else
            echo "Using ASR-reported audio duration: $AUDIO_DURATION seconds" | tee -a $LOG_FILE
        fi
        
        # Analyze power consumption if tegrastats data is available
        POWER_METRICS="$METRICS_DIR/power_metrics_${model_name}_${precision}_$CURRENT_DATETIME.json"
        if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
            echo "Analyzing power consumption..." | tee -a $LOG_FILE
            MODEL_FAMILY=$(get_model_family "$model")
            
            if python src/analysis/power.py "$OUTPUT_FILE" \
                --precision "$precision" \
                --model_family "$MODEL_FAMILY" \
                --audio_duration "$AUDIO_DURATION" \
                --metrics_file "$METRICS_JSON" \
                --output_json "$POWER_METRICS" \
                --verbose 2>&1 | tee -a $LOG_FILE; then
                echo "Power analysis completed successfully" | tee -a $LOG_FILE
            else
                echo "Warning: Power analysis failed, continuing without power metrics" | tee -a $LOG_FILE
            fi
        else
            echo "Skipping power analysis (no tegrastats data)" | tee -a $LOG_FILE
        fi
        
        # Calculate Word Error Rate
        echo "Calculating Word Error Rate..." | tee -a $LOG_FILE
        WER_JSON="$METRICS_DIR/wer_results_${model_name}_${precision}.json"
        
        if python src/analysis/wer.py "$OUTPUT_DIR" \
            --reference_file "$REFERENCE_FILE" \
            --output_json "$WER_JSON" \
            --details --verbose 2>&1 | tee -a $LOG_FILE; then
            
            # Extract WER from JSON output
            WER=$(python3 -c "
import json
try:
    with open('$WER_JSON', 'r') as f:
        data = json.load(f)
    print(f\"{data.get('wer_percent', 0.0):.2f}\")
except:
    print('0.00')
" 2>/dev/null || echo "0.00")
            echo "WER calculation completed: $WER%" | tee -a $LOG_FILE
        else
            echo "Warning: WER calculation failed, using default value" | tee -a $LOG_FILE
            WER="0.00"
        fi
        
        # Update power metrics with WER if both files exist
        if [ -f "$POWER_METRICS" ] && [ -f "$WER_JSON" ]; then
            python3 -c "
import json
try:
    # Load power metrics
    with open('$POWER_METRICS', 'r') as f:
        power_data = json.load(f)
    
    # Load WER results
    with open('$WER_JSON', 'r') as f:
        wer_data = json.load(f)
    
    # Merge data
    power_data.update(wer_data)
    
    # Save combined metrics
    with open('$POWER_METRICS', 'w') as f:
        json.dump(power_data, f, indent=2)
    
    print('Metrics successfully merged')
except Exception as e:
    print(f'Error merging metrics: {e}')
" 2>&1 | tee -a $LOG_FILE
        fi
        
        # Calculate Real-Time Factor (RTF)
        RTF=$(echo "scale=3; $EXEC_TIME_SEC / $AUDIO_DURATION" | bc -l 2>/dev/null || echo "0.000")
        
        # Extract additional metrics from power analysis
        TOTAL_ENERGY_JOULE="0.0"
        EPAS="0.0"
        HUR="0.0"
        GME="0.0"
        MEAN_VALUE="0.0"
        MAX_VALUE="0.0"
        
        if [ -f "$POWER_METRICS" ]; then
            TOTAL_ENERGY_JOULE=$(python3 -c "
import json
try:
    with open('$POWER_METRICS', 'r') as f:
        data = json.load(f)
    print(data.get('TOTAL_ENERGY_JOULE', 0.0))
except:
    print(0.0)
" 2>/dev/null || echo "0.0")
            
            EPAS=$(python3 -c "
import json
try:
    with open('$POWER_METRICS', 'r') as f:
        data = json.load(f)
    print(data.get('EPAS', 0.0))
except:
    print(0.0)
" 2>/dev/null || echo "0.0")
            
            HUR=$(python3 -c "
import json
try:
    with open('$POWER_METRICS', 'r') as f:
        data = json.load(f)
    print(data.get('HUR', 0.0))
except:
    print(0.0)
" 2>/dev/null || echo "0.0")
            
            GME=$(python3 -c "
import json
try:
    with open('$POWER_METRICS', 'r') as f:
        data = json.load(f)
    print(data.get('GME', 0.0))
except:
    print(0.0)
" 2>/dev/null || echo "0.0")
            
            MEAN_VALUE=$(python3 -c "
import json
try:
    with open('$POWER_METRICS', 'r') as f:
        data = json.load(f)
    print(data.get('MEAN_MEMORY_MB', 0.0))
except:
    print(0.0)
" 2>/dev/null || echo "0.0")
            
            MAX_VALUE=$(python3 -c "
import json
try:
    with open('$POWER_METRICS', 'r') as f:
        data = json.load(f)
    print(data.get('MAX_MEMORY_MB', 0.0))
except:
    print(0.0)
" 2>/dev/null || echo "0.0")
        fi
        
        # Log comprehensive results
        LOG_ENTRY="Model: $model, Precision: $precision, Family: $(get_model_family $model)"
        LOG_ENTRY="$LOG_ENTRY, Execution Time: $EXEC_TIME ($EXEC_TIME_SEC sec), RTF: $RTF"
        LOG_ENTRY="$LOG_ENTRY, WER: $WER%, Audio Duration: $AUDIO_DURATION sec"
        LOG_ENTRY="$LOG_ENTRY, Total Energy (J): $TOTAL_ENERGY_JOULE, EPAS: $EPAS"
        LOG_ENTRY="$LOG_ENTRY, Mean Mem (MB): $MEAN_VALUE, Max Mem (MB): $MAX_VALUE"
        LOG_ENTRY="$LOG_ENTRY, GME: $GME, HUR: $HUR"
        
        echo "$LOG_ENTRY" >> "$LOG_FILE"
        echo "Experiment completed successfully for $model with $precision precision" | tee -a $LOG_FILE
        echo "Results saved to: $METRICS_DIR" | tee -a $LOG_FILE
        
    done
done

echo "========================================" | tee -a $LOG_FILE
echo "All model evaluations completed!" | tee -a $LOG_FILE
echo "Consolidating metrics..." | tee -a $LOG_FILE

# Consolidate all metrics into a single CSV
if python metrics_collector.py --log_file "$LOG_FILE" --metrics_dir "$METRICS_BASE_DIR" --output_csv asr_metrics_results.csv 2>&1 | tee -a $LOG_FILE; then
    echo "Metrics consolidation completed successfully" | tee -a $LOG_FILE
else
    echo "Warning: Metrics consolidation failed" | tee -a $LOG_FILE
fi

# Generate summary report
echo "=== EVALUATION SUMMARY ===" | tee -a $LOG_FILE
echo "Completed at: $(date)" | tee -a $LOG_FILE
echo "Total models evaluated: ${#models[@]}" | tee -a $LOG_FILE
echo "Precision settings: ${precisions[*]}" | tee -a $LOG_FILE
echo "Total experiments: $((${#models[@]} * ${#precisions[@]}))" | tee -a $LOG_FILE
echo "Total audio duration processed: $TOTAL_AUDIO_DURATION seconds" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Output files:" | tee -a $LOG_FILE
echo "- Detailed log: $LOG_FILE" | tee -a $LOG_FILE
echo "- Consolidated metrics: asr_metrics_results.csv" | tee -a $LOG_FILE
echo "- Individual results: $METRICS_BASE_DIR/" | tee -a $LOG_FILE
echo "- Transcription outputs: $OUTPUT_BASE_DIR/" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "To generate green scores with different weighting schemes:" | tee -a $LOG_FILE
echo "python crc/utils/asr_metrics_aggregator.py asr_metrics_results.csv --schemes balanced,mobile,realtime,server" | tee -a $LOG_FILE

echo "ASR Green Metrics Framework evaluation completed successfully!"
echo "Check $LOG_FILE for detailed logs and asr_metrics_results.csv for consolidated results."