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
pip install transformers
pip install numpy
pip install soundfile
pip install librosa
pip install onnxruntime
pip install matplotlib
conda install -c conda-forge opencv=4.8.0 ffmpeg=6.1.1
```

## 2. Video Preprocessing

Download Kanghui demo video at https://www.bilibili.com/video/av885252652/

```bash
cd data_utils
# Convert video to 20fps with better quality using libx264
ffmpeg -i kanghui.mp4 -filter:v fps=20 -c:v libx264 kanghui_20fps_better.mp4
python3 ../crop_video.py kanghui_20fps_better.mp4 kanghui_cropped.mp4
ffmpeg -i kanghui_cropped.mp4 -i kanghui_20fps_better.mp4 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 kanghui_cropped_with_audio.mp4
# Extract 5-minute segment from video
ffmpeg -ss 300 -i kanghui_cropped_with_audio.mp4 -t 300 -c:v copy -c:a copy kanghui_5min_segment.mp4

# Split video into training (70%) and validation (30%) sets
ffmpeg -i kanghui_cropped_with_audio.mp4 -t 00:15:51 -c copy kanghui_train.mp4 && ffmpeg -i kanghui_cropped_with_audio.mp4        -ss 00:15:51        -c:a copy        -c:v libx264 -preset medium -crf 23  kanghui_val.mp4
```

## 3. Preprocessing

```bash
# Create directories for training and validation data
mkdir -p train_data val_data

# Preprocess training video
python process.py ./kanghui_train.mp4 --asr wenet && mv full_body_img landmarks aud.wav aud_wenet.npy train_data/

# Preprocess validation video
python process.py ./kanghui_val.mp4 --asr wenet && mv full_body_img landmarks aud.wav aud_wenet.npy val_data/
```

## 4. Training

```bash
# Train the model with WeNet features
# python train.py --dataset_dir ./data_utils --save_dir ./checkpoint/ --asr wenet

# Train with validation set and plot loss
python train.py --dataset_dir ./data_utils/train_data --val_dataset_dir ./data_utils/val_data --save_dir ./checkpoint/ --asr wenet --plot_loss
```

## 5. Inference

```bash
cd data_utils
python wenet_infer.py seedtts-01_zh.wav 
# Run inference with WeNet features
python ../inference.py --asr=wenet --audio_feat=seedtts-01_zh_wenet.npy --save_path=test.mp4 --checkpoint=..
/checkpoint/135.pth --dataset=./
```

## 6. Video and Audio Combination

```bash
# Combine video and audio using libopenh264 codec
ffmpeg -i test.mp4 -i seedtts-01_zh.wav -c:v libopenh264 -c:a aac result_test.mp4
```

## Notes
- The training process showed successful convergence with loss values decreasing from ~0.02 to ~0.007 over 200 epochs
- Make sure to use the correct file paths and extensions when running commands
- Audio quality is crucial for good results - use external microphones and avoid noisy environments 