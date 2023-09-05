import cv2
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Display video frames with annotated rectangles')
    parser.add_argument('--vid_path', type=str, help='Path to the video file')
    parser.add_argument('--annot_path', type=str, help='Path to the CSV annotations file')
    parser.add_argument('--output_path', type=str, default='output_video.avi', help='Path to the output video file')
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame number')
    parser.add_argument('--num_frames', type=int, default=1000, help='Number of frames to display')
    args = parser.parse_args()
    
    annotations = pd.read_csv(args.annot_path)
    start_frame = args.start_frame
    cap = cv2.VideoCapture(args.vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
    
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (frame_width, frame_height))
    
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        else:
            counter = counter + 1
            frame_num = args.start_frame + counter
            frame_annotations = annotations[annotations['frame_num'] == frame_num]
            
            if frame_annotations.empty:
                continue
            else:
                max_confidence_row = frame_annotations.loc[frame_annotations['confidence'].idxmax()]
                rect_coords = (int(max_confidence_row['xmin']), int(max_confidence_row['ymin']), int(max_confidence_row['xmax']), int(max_confidence_row['ymax']))
                cv2.rectangle(frame, (rect_coords[0], rect_coords[1]), (rect_coords[2], rect_coords[3]), (0, 255, 0), 2)
                
            counter_text = f'frame: {frame_num}'
            cv2.putText(frame, counter_text, (150,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)            
            cv2.imshow('Video Frame', frame)
            out.write(frame)
            
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            
            if counter >= args.num_frames:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


