# Talk Is Cheap, Energy Is Not: Towards a Green, Context-Aware Metrics Framework for Automatic Speech Recognition

A comprehensive multimetric framework for evaluating Automatic Speech Recognition (ASR) models with a focus on both accuracy and efficiency ("green") metrics.

This release corresponds to the paper: "Talk Is Cheap, Energy Is Not: Towards a Green, Context-Aware Metrics Framework for Automatic Speech Recognition" accepted at ECML PKDD 2025, Applied Data Science Track.

## Overview

This framework allows you to benchmark various ASR models available through Hugging Face, evaluating not just their transcription accuracy, but also their computational efficiency, energy consumption, and hardware utilization. The framework is designed to help determine which models are most suitable for different deployment scenarios (server-side, mobile/edge, real-time applications).

## Features

- Evaluation of multiple ASR model architectures (Whisper, Wav2Vec2, HuBERT, WavLM, UniSpeech, SpeechT5)
- Support for different precision settings (FP16, FP32)
- Multi-format audio support (FLAC, MP3) and transcriptions support (txt, csv)
- Platform-aware power analysis for Jetson devices
- Comprehensive metrics collection:
  - Word Error Rate (WER) for transcription accuracy
  - Real-Time Factor (RTF) for processing speed
  - Energy Per Audio Second (EPAS) for power efficiency
  - 95th percentile latency measurements
  - GPU Memory Efficiency (GME) for memory utilization
  - Hardware Utilization Ratio (HUR) for CPU/GPU efficiency
- Configurable weighting schemes for different deployment scenarios
- Robust error handling and detailed logging
- Automatic report generation with metrics visualization

## Requirements

- Python 3.7+
- PyTorch with CUDA support
- NVIDIA GPU with CUDA compatibility
- NVIDIA Jetson device with tegrastats support (for power measurements)
- soundfile, jiwer, numpy, pandas, matplotlib, seaborn

## Installation

1. Install dependencies:
   ```
   pip install torch torchaudio transformers soundfile jiwer numpy pandas matplotlib seaborn
   ```

2. Place your audio files in the `./data` directory with corresponding reference transcriptions in `./data/trans.txt`

## Usage

### Basic Usage

Run the full benchmark suite with default settings:

```bash
./run.sh
```

This will:
- Run all supported ASR models in both FP16 and FP32 precision
- Process all audio files in the `./data` directory
- Collect performance and efficiency metrics
- Calculate Word Error Rates
- Generate metrics in `asr_metrics_results.csv`

### Custom Model Evaluation

To evaluate a specific model with custom settings:

```bash
# Example for running Wav2Vec2 model
python wav2vec2.py facebook/wav2vec2-large-960h fp16 ./data ./output/wav2vec2_fp16 --metrics_file metrics_wav2vec2_fp16.json
```

### Metrics Aggregation

To calculate green scores with different weighting schemes:

```bash
python asr_metrics_aggregator.py asr_metrics_results.csv --schemes balanced,mobile,realtime,server
```

## Metrics and Scoring

The framework calculates a composite "green score" based on the following metrics:

- **WER (Word Error Rate)**: Accuracy of transcription (lower is better)
- **RTF (Real-Time Factor)**: Processing speed relative to audio duration (lower is better)
- **EPAS (Energy Per Audio Second)**: Power efficiency in Joules/second (lower is better)
- **Latency**: 95th percentile response time in milliseconds (lower is better)
- **GME (Green Memory Efficiency)**: Memory utilization efficiency (higher is better)
- **HUR (Hardware Utilization Ratio)**: CPU/GPU utilization efficiency, optimal around 60-90%

### Weighting Schemes

The framework provides pre-defined weighting schemes for different scenarios:

1. **Balanced**: General purpose weighting that balances all aspects
2. **Mobile Edge**: Optimized for battery-powered and resource-constrained devices
3. **Real-Time**: Optimized for voice assistants and interactive systems
4. **Server-Side**: Optimized for large-scale cloud deployments
5. **Custom**: User-defined weighting

## File Structure

- **Model runner scripts**:
  - `whisper.py`: Runner for Distil-Whisper models
  - `wav2vec2.py`: Runner for Wav2Vec2 models
  - `hubert.py`: Runner for HuBERT models
  - `wavLM.py`: Runner for WavLM models
  - `unispeech.py`: Runner for UniSpeech models
  - `speecht5.py`: Runner for SpeechT5 models

- **Metrics collection**:
  - `power.py`: Analyzes power consumption from tegrastats output
  - `wer.py`: Calculates Word Error Rate with sophisticated text normalization
  - `metrics_collector.py`: Consolidates metrics from multiple sources

- **Metrics aggregation**:
  - `asr_metrics_aggregator.py`: Calculates green scores with configurable weights

- **Orchestration**:
  - `run.sh`: Main script to run the complete benchmark suite

## Example Output

After running the benchmark, you will get:

1. Transcription files for each audio input in the `./output` directory
2. Individual metrics JSON files in `./metrics_data`
3. A consolidated CSV file `asr_metrics_results.csv` with all metrics
4. Green score visualizations in the `./weighted_metrics` directory

## Extending the Framework

### Adding New Models

1. Create a new runner script based on an existing one (e.g., `model.py`)
2. Update the `run.sh` script to include your new model
3. Add any model-specific configurations to the relevant metrics scripts 

### Adding New Metrics

1. Update the relevant model runner to collect the new metric
2. Modify `metrics_collector.py` to include the new metric
3. Update `asr_metrics_aggregator.py` to include the metric in scoring calculations

## License

MIT 

## Acknowledgements
The "Talk Is Cheap, Energy Is Not: Towards a Green, Context-Aware Metrics Framework for Automatic Speech Recognition" paper is supported by the European Union’s HORIZON Research and Innovation Programme under grant agreement No. 101120657, project ENFIELD (European Lighthouse to Manifest Trustworthy and Green AI).
