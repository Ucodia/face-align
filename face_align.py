import os
import cv2
import dlib
import numpy as np
import scipy.ndimage
import PIL.Image
import urllib.request
import bz2
from pathlib import Path
import sys

def get_predictor_path():
    """Get the system-standard path for storing the predictor file."""
    if os.name == 'posix':
        if sys.platform == 'darwin':  # macOS
            base_dir = Path.home() / 'Library' / 'Application Support' / 'face_align'
        else:  # Linux
            base_dir = Path.home() / '.local' / 'share' / 'face_align'
    else:  # Windows
        base_dir = Path(os.getenv('APPDATA')) / 'face_align'
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / 'shape_predictor_68_face_landmarks.dat')

def download_predictor_model(predictor_path):
    print("Downloading shape predictor file...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_path = predictor_path + ".bz2"
    urllib.request.urlretrieve(url, bz2_path)
    with bz2.BZ2File(bz2_path, 'rb') as source, open(predictor_path, 'wb') as target:
        target.write(source.read())
    os.remove(bz2_path)
    print(f"Shape predictor downloaded and extracted successfully to {predictor_path}")

def align_image(image, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True):
    img = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if isinstance(image, np.ndarray) else image)

    # calculate vectors
    lm = np.array(face_landmarks)
    eye_left     = np.mean(lm[36 : 42], axis=0)
    eye_right    = np.mean(lm[42 : 48], axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_avg = (lm[48] + lm[54]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # rotate crop rectangle
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c0 = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
    qsize = np.hypot(*x) * 2

    # shrink
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # crop
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # pad
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        img = PIL.Image.fromarray(img, 'RGB')
        quad += pad[:2]

    # transform
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS)

    # Convert back to numpy array in BGR format for OpenCV
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class FaceAlign:
    def __init__(self, output_size=1024):
        predictor_model_path = get_predictor_path()
        if not os.path.exists(predictor_model_path):
            download_predictor_model(predictor_model_path)
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)
        self.output_size = output_size
    def get_aligned_image(self, image):
        lms = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(gray, detection).parts()]
            lms.append(face_landmarks)

        if len(lms) < 1:
            return None
        out_image = align_image(image, lms[0], output_size=self.output_size)
        return out_image
