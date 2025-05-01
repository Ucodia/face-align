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
import mediapipe as mp
from PIL import ImageDraw

MP_EYE_LEFT_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
MP_EYE_RIGHT_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
MP_MOUTH_IDX = [61, 291]
DLIB_EYE_LEFT_IDX = list(range(36, 42))
DLIB_EYE_RIGHT_IDX = list(range(42, 48))
DLIB_MOUTH_IDX = [48, 54]

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

def align_image(image, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True, debug=False):
    img = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if isinstance(image, np.ndarray) else image)

    # calculate vectors
    lm = np.array(face_landmarks)

    if (len(lm) == 468):
        eye_left_idx = MP_EYE_LEFT_IDX
        eye_right_idx = MP_EYE_RIGHT_IDX
        mouth_idx = MP_MOUTH_IDX
    elif (len(lm) == 68):
        eye_left_idx = DLIB_EYE_LEFT_IDX
        eye_right_idx = DLIB_EYE_RIGHT_IDX
        mouth_idx = DLIB_MOUTH_IDX
    else:
        raise ValueError(f"Unsupported number of landmarks: {len(lm)}")
    
    eye_left_mean = np.mean(lm[eye_left_idx], axis=0)
    eye_right_mean = np.mean(lm[eye_right_idx], axis=0)
    mouth_mean = np.mean(lm[mouth_idx], axis=0)

    eye_avg = (eye_left_mean + eye_right_mean) * 0.5
    eye_to_eye = eye_right_mean - eye_left_mean
    eye_to_mouth = mouth_mean - eye_avg

    if debug:
        # overlay debug information
        draw = ImageDraw.Draw(img)
        key_points = eye_left_idx + eye_right_idx + mouth_idx
        radius = int(np.linalg.norm(eye_to_eye) * 0.02)
        for idx in key_points:
            x, y = lm[idx]
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(0,255,0))

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
    def __init__(self, output_size=1024, engine='mediapipe', debug=False):
        """
        Initialize the face alignment class.
        
        Args:
            output_size (int): Size of the output aligned face image
            engine (str): Type of face detection engine to use ('mediapipe' or 'dlib')
            debug (bool): Whether to show landmarks in the output image
        """
        self.output_size = output_size
        self.engine = engine.lower()
        self.debug = debug
        
        if self.engine == 'mediapipe':
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
        elif self.engine == 'dlib':
            predictor_model_path = get_predictor_path()
            if not os.path.exists(predictor_model_path):
                download_predictor_model(predictor_model_path)
            self.detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor(predictor_model_path)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}.")

    def get_face_landmarks(self, image):
        """
        Get face landmarks using the specified engine.
        
        Args:
            image: Input image (numpy array in BGR format)
            
        Returns:
            Face landmarks or None if no face is detected
        """
        if self.engine == 'mediapipe':
            # Convert the BGR image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and get face landmarks
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
                
            # Get the first face landmarks
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Convert landmarks to pixel coordinates
            h, w = image.shape[:2]
            return [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
            
        elif self.engine == 'dlib':
            # Convert to grayscale for dlib
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            dets = self.detector(gray, 1)
            
            if len(dets) < 1:
                return None
                
            # Get landmarks for the first detected face
            shape = self.shape_predictor(gray, dets[0])
            return [(item.x, item.y) for item in shape.parts()]

    def get_aligned_image(self, image):
        """
        Get an aligned face image using the specified engine.
        
        Args:
            image: Input image (numpy array in BGR format)
            
        Returns:
            Aligned face image or None if no face is detected
        """
        landmarks = self.get_face_landmarks(image)
        # Align the image using the detected landmarks
        return align_image(image, landmarks, output_size=self.output_size, debug=self.debug)
