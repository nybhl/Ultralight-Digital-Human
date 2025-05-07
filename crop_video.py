import cv2
import numpy as np
import argparse

def get_average_face_position(video_path, sample_frames=100):
    """
    Calculate average face position by sampling frames from the video
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly throughout the video
    sample_indices = np.linspace(0, total_frames-1, min(sample_frames, total_frames), dtype=int)
    
    face_positions = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Calculating average face position...")
    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            center_x = x + w // 2
            center_y = y + h // 2
            face_positions.append((center_x, center_y, w, h))
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sample_indices)} sample frames")
    
    cap.release()
    
    if not face_positions:
        raise ValueError("No faces detected in the video")
    
    # Calculate average position and size
    avg_center_x = int(np.mean([p[0] for p in face_positions]))
    avg_center_y = int(np.mean([p[1] for p in face_positions]))
    avg_w = int(np.mean([p[2] for p in face_positions]))
    avg_h = int(np.mean([p[3] for p in face_positions]))
    
    return avg_center_x, avg_center_y, avg_w, avg_h

def crop_video(input_path, output_path):
    """
    Crop video to focus on head and body area with dimensions proportional to head size
    """
    # Get average face position and size
    try:
        avg_center_x, avg_center_y, avg_w, avg_h = get_average_face_position(input_path)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Calculate crop dimensions (twice the head size)
    width = avg_w * 2
    height = avg_h * 2
    
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {input_path}")
    print(f"Total frames: {total_frames}")
    print(f"Using fixed crop center: ({avg_center_x}, {avg_center_y})")
    print(f"Crop dimensions: {width}x{height} (2x head size)")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate crop coordinates using average position
        crop_x1 = max(0, avg_center_x - width // 2)
        crop_y1 = max(0, avg_center_y - height // 2)
        crop_x2 = min(frame.shape[1], crop_x1 + width)
        crop_y2 = min(frame.shape[0], crop_y1 + height)
        
        # Adjust if we hit the edges
        if crop_x2 - crop_x1 < width:
            if crop_x1 == 0:
                crop_x2 = width
            else:
                crop_x1 = crop_x2 - width
        if crop_y2 - crop_y1 < height:
            if crop_y1 == 0:
                crop_y2 = height
            else:
                crop_y1 = crop_y2 - height
        
        # Crop and resize
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        resized = cv2.resize(cropped, (width, height))
        
        # Write frame
        out.write(resized)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Release everything
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop video to focus on head and body area')
    parser.add_argument('input_path', type=str, help='Path to input video file')
    parser.add_argument('output_path', type=str, help='Path to save output video file')
    parser.add_argument('--sample_frames', type=int, default=100, help='Number of frames to sample for average position (default: 100)')
    
    args = parser.parse_args()
    crop_video(args.input_path, args.output_path) 