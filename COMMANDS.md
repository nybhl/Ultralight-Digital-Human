# Ultralight Digital Human - Command History

This document contains all the successful commands used in chronological order for the Ultralight Digital Human project.

## 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n dh python=3.10
conda activate dh

# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
conda install mkl
pip install opencv-python
pip install transformers
pip install numpy
pip install soundfile
pip install librosa
pip install onnxruntime
```

## 2. Video Preprocessing

```bash
# Convert video to 20fps with better quality using libopenh264
ffmpeg -i kanghui.mp4 -filter:v fps=20 -c:v libopenh264 kanghui_20fps_better.mp4

# Extract 5-minute segment from video
ffmpeg -ss 300 -i kanghui_20fps_better.mp4 -t 300 -c:v copy -c:a copy kanghui_5min_segment.mp4
```

## 3. Preprocessing

```bash
# Run data preprocessing
cd data_utils
python process.py ../kanghui_20fps_better.mp4 --asr wenet
```

## 4. Training

```bash
# Train the model with WeNet features
python train.py --dataset_dir . --save_dir ./checkpoint/ --asr wenet
```

## 5. Inference

```bash
# Run inference with WeNet features
python inference.py --asr=wenet --audio_feat=aud_wenet.npy --save_path=test.mp4 --checkpoint=./checkpoint/195.pth --dataset=./
```

## 6. Video and Audio Combination

```bash
# Combine video and audio using libopenh264 codec
ffmpeg -i test.mp4 -i aud.wav -c:v libopenh264 -c:a aac result_test.mp4
```

## Notes
- The training process showed successful convergence with loss values decreasing from ~0.02 to ~0.007 over 200 epochs
- For video encoding, use `libopenh264` instead of `libx264` as the latter is not available in the current FFmpeg installation
- Make sure to use the correct file paths and extensions when running commands
- Audio quality is crucial for good results - use external microphones and avoid noisy environments 