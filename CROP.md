# Video Cropping Commands

This document describes how to use the video cropping script to create passport-style head and body crops of videos.

## Basic Usage

To crop a video with fixed dimensions (640x640):

```bash
python crop_video.py input_video.mp4 output_video.mp4
```

## Proportional Cropping

To crop a video with dimensions proportional to the detected head size (2x head width and height):

```bash
python crop_video.py input_video.mp4 output_video.mp4
```

The script will automatically:
1. Sample frames from the video to detect the average head position and size
2. Calculate crop dimensions as twice the head size
3. Apply the same crop position and size to all frames

## Adding Audio

The cropping script doesn't preserve audio. To add audio from the original video to the cropped video:

```bash
ffmpeg -i cropped_video.mp4 -i original_video.mp4 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 output_with_audio.mp4
```

For example:
```bash
ffmpeg -i data_utils/kanghui_cropped_proportional.mp4 -i data_utils/kanghui.mp4 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 data_utils/kanghui_cropped_proportional_with_audio.mp4
```

## Parameters

- `input_video.mp4`: Path to the input video file
- `output_video.mp4`: Path where the cropped video will be saved
- `--sample_frames`: Number of frames to sample for calculating average position (default: 100)

## Example

To crop Kanghui's video with proportional dimensions:

```bash
python crop_video.py data_utils/kanghui.mp4 data_utils/kanghui_cropped.mp4
```

## How It Works

1. The script first analyzes multiple frames to find the average face position and size
2. It then uses this fixed position to crop all frames consistently
3. The crop dimensions are set to twice the detected head size
4. If the crop would go beyond the video boundaries, it's adjusted to stay within the frame

## Output

The script will print:
- The total number of frames in the video
- The fixed crop center coordinates
- The crop dimensions
- Progress updates every 100 frames

## Notes

- The script uses OpenCV's face detection
- The output video maintains the original frame rate
- The crop position is fixed throughout the video for stability
- The crop size is proportional to the detected head size for a natural look
- The cropped video will not have audio - you need to add it separately using ffmpeg 