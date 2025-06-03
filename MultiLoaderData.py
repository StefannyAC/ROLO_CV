import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from ultralytics import YOLO
from YOLOv8FeatureExtractor import YOLOv8FeatureExtractor

class OTBMultiSequenceDataset(Dataset):
    def __init__(self, root_dir, extractor, seq_len=10, image_size=(640, 640), device='cuda'):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.image_size = image_size
        self.device = device
        self.extractor = extractor

        self.extractor = YOLOv8FeatureExtractor(model_path='yolov8s.pt', device=self.device)

        self.samples = []  # Cada elemento: (video_path, frame_start_idx)

        self.video_info = []  # Almacena info por carpeta: (frames_list, gt_boxes)

        list_path = os.path.join(root_dir, 'otb30_list.txt')
        with open(list_path, 'r') as f:
            selected_dirs = [line.strip() for line in f.readlines() if line.strip()]

        video_dirs = [d for d in selected_dirs if os.path.isdir(os.path.join(root_dir, d))]

        for video_name in video_dirs:
            video_path = os.path.join(root_dir, video_name)
            frame_paths = sorted(glob.glob(os.path.join(video_path, 'img', '*.jpg')))
            gt_path = os.path.join(video_path, 'groundtruth_rect.txt')
            if not os.path.exists(gt_path) or len(frame_paths) == 0:
                continue

            gt_boxes = np.loadtxt(gt_path, delimiter=',')

            if len(gt_boxes) != len(frame_paths):
                continue  # skip inconsistent videos

            self.video_info.append((frame_paths, gt_boxes))

            # Generar muestras por secuencia
            for i in range(seq_len - 1, len(frame_paths)):
                self.samples.append((len(self.video_info) - 1, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_idx, frame_end = self.samples[idx]
        frame_paths, gt_boxes = self.video_info[video_idx]

        start = frame_end - self.seq_len + 1
        imgs = []
        yolo_feats = []
        norm_boxes = []

        for i in range(start, frame_end + 1):
            img = cv2.imread(frame_paths[i])
            img_resized = cv2.resize(img, self.image_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            _, vector, _ = self.extractor.extract(img_rgb)
            yolo_feats.append(torch.tensor(vector, dtype=torch.float32))

            x, y, w, h = gt_boxes[i]
            H, W = self.image_size
            norm_box = torch.tensor([x/W, y/H, w/W, h/H], dtype=torch.float32)
            norm_boxes.append(norm_box)

        inputs = [torch.cat([feat, box], dim=0) for feat, box in zip(yolo_feats, norm_boxes)]
        inputs = torch.stack(inputs)
        target_box = norm_boxes[-1]

        return inputs, target_box
