import cv2
import numpy as np
import argparse
from tqdm import tqdm
import mediapipe as mp

def process_frame(frame, selfie_segmentation):
    """
    Process a single frame to change background to white while preserving human using MediaPipe
    """
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get the segmentation mask
    results = selfie_segmentation.process(rgb_frame)
    
    if not results.segmentation_mask is None:
        # Convert the mask to binary
        mask = (results.segmentation_mask > 0.1).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Create a white background
        white_bg = np.ones_like(frame) * 255
        
        # Use the mask to combine the original frame and white background
        result = cv2.bitwise_and(frame, frame, mask=mask)
        inverse_mask = cv2.bitwise_not(mask)
        white_bg = cv2.bitwise_and(white_bg, white_bg, mask=inverse_mask)
        result = cv2.add(result, white_bg)
        
        return result
    else:
        return frame

def change_background(input_path, output_path):
    """
    Change video background to white while preserving human using MediaPipe
    """
    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1  # 1 for general model, 0 for landscape model
    )
    
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {input_path}")
    print(f"Total frames: {total_frames}")
    
    # Process each frame
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        processed_frame = process_frame(frame, selfie_segmentation)
        
        # Write the processed frame
        out.write(processed_frame)
    
    # Release everything
    cap.release()
    out.release()
    selfie_segmentation.close()
    print(f"Video processing complete. Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Change video background to white while preserving human using MediaPipe')
    parser.add_argument('input_path', type=str, help='Path to input video file')
    parser.add_argument('output_path', type=str, help='Path to save output video file')
    
    args = parser.parse_args()
    change_background(args.input_path, args.output_path) 