#!/usr/bin/python3

import cv2
import argparse

def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def main():
    parser = argparse.ArgumentParser(description='Count total frames in a video')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    args = parser.parse_args()

    total_frames = get_total_frames(args.video_path)
    if total_frames is not None:
        print(f"Total frames in the video: {total_frames}")

if __name__ == '__main__':
    main()
