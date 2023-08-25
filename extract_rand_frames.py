import cv2
import random
import argparse
import random
import os

# Set a specific seed for reproducibility
seed_value = 42
random.seed(seed_value)


# Argument parsing
parser = argparse.ArgumentParser(description="Extract N random frames from a video for RoboFlow workflow")
parser.add_argument("trial_name", help="Name of the trial")
args = parser.parse_args()


# Create and specify a directory to save the frames
output_directory = args.trial_name +  '/random_frames/'
os.makedirs(output_directory, exist_ok=True)

# Number of random frames to extract
N = 50


# Specify the cropping coordinates (x, y, width, height) for the ROI
crop_x = 800
crop_y = 0
crop_width = 2200
crop_height = 2200

# Specify the desired output frame size
output_width = 640
output_height = 640

# Open the video
video_path = args.trial_name + '/Cam0.mp4'
cap = cv2.VideoCapture(video_path)

# Get the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_offset = 7200 # remove these many frames from start and end

# Generate N random frame indices
random_frame_indices = random.sample(range(frame_offset,total_frames-frame_offset), N)

counter = 0

# Read and process random frames
for frame_index in random_frame_indices:
    
    counter = counter + 1
    
    # Set the current frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    
    if ret:
        
        # Crop the frame to the specified ROI
        cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    
        # Resize the cropped frame to the desired output size
        resized_frame = cv2.resize(cropped_frame, (output_width, output_height))
        
        # Save the frame to the specified directory
        frame_filename = os.path.join(output_directory, f'frame_{frame_index}.png')
        cv2.imwrite(frame_filename, resized_frame)
        print(f"Saved frame {counter} / {N} :  {frame_index} to {frame_filename}")
        
        # # Process the frame (e.g., display, save, etc.)
        # cv2.imshow(f'Random Frame {frame_index}', frame)
        # cv2.waitKey(0)

# Release the video capture
cap.release()
cv2.destroyAllWindows()
