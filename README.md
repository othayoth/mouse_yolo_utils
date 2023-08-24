# mouse_yolo_utils

### Training
Might have to change the relative path in the dataset's `data.yaml` so that path to train/test/valid images are correct

### Detection
```
python detect.py --weights runs/train/exp19/weights/best.pt --save-txt --save-conf --source ~/data_storage/maya_pd_videos/23_08_04/ctrl2/ctrl2_cropped.mp4 

```
