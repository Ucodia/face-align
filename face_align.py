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
import argparse
from tqdm import tqdm

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

    if (len(lm) == 468 or len(lm) == 478):  # +10 landmarks when enabling refine_landmarks option
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
    def __init__(self, output_size=1024, engine='mediapipe', debug=False, max_faces=3, refine_landmarks=False):
        """
        Initialize the face alignment class.
        
        Args:
            output_size (int): Size of the output aligned face image
            engine (str): Type of face detection engine to use ('mediapipe' or 'dlib')
            debug (bool): Whether to show landmarks in the output image
            max_faces (int): Maximum number of faces to detect
            refine_landmarks (bool): Whether to enable refined landmarks in MediaPipe
        """
        self.output_size = output_size
        self.engine = engine.lower()
        self.debug = debug
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        
        if self.engine == 'mediapipe':
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_faces,
                refine_landmarks=refine_landmarks,
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
            List of face landmarks (one per detected face), empty list if no faces are detected
        """
        if self.engine == 'mediapipe':
            # Convert the BGR image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and get face landmarks
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return []
                
            # Get all face landmarks
            all_face_landmarks = []
            h, w = image.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks to pixel coordinates
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                all_face_landmarks.append(landmarks)
            
            return all_face_landmarks
            
        elif self.engine == 'dlib':
            # Convert to grayscale for dlib
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            dets = self.detector(gray, 1)
            
            if len(dets) < 1:
                return []
                
            # Get landmarks for all detected faces
            all_face_landmarks = []
            for det in dets:
                shape = self.shape_predictor(gray, det)
                landmarks = [(item.x, item.y) for item in shape.parts()]
                all_face_landmarks.append(landmarks)
            
            return all_face_landmarks

    def get_aligned_images(self, image, max_faces=None):
        """
        Get aligned face images for detected faces using the specified engine.
        
        Args:
            image: Input image (numpy array in BGR format)
            max_faces: Maximum number of faces to process (None for all faces)
            
        Returns:
            List of aligned face images, empty list if no faces are detected
        """
        landmarks_list = self.get_face_landmarks(image)
        if not landmarks_list:
            return []
        
        # Limit the number of faces to process if specified
        if max_faces is not None:
            landmarks_list = landmarks_list[:max_faces]
        
        # Align the image for each detected face
        aligned_images = []
        for landmarks in landmarks_list:
            aligned_image = align_image(image, landmarks, output_size=self.output_size, debug=self.debug)
            aligned_images.append(aligned_image)
        
        return aligned_images

def process_image(input_path, output_path, output_size, engine='mediapipe', debug=False, max_faces=3, refine_landmarks=False):
    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image '{input_path}'")
            return False

        face_aligner = FaceAlign(output_size=output_size, engine=engine, debug=debug, max_faces=max_faces, refine_landmarks=refine_landmarks)
        aligned_images = face_aligner.get_aligned_images(image, max_faces=max_faces)
        
        if not aligned_images:
            print(f"Error: No face detected in image '{input_path}'")
            return False
        
        # Save each aligned face with appropriate naming
        output_path_obj = Path(output_path)
        success_count = 0
        
        for i, aligned_image in enumerate(aligned_images):
            if len(aligned_images) == 1:
                # Single face, use original output path
                face_output_path = str(output_path_obj)
            else:
                # Multiple faces, add face index suffix
                face_output_path = str(output_path_obj.with_name(
                    f"{output_path_obj.stem}_face{i+1:02d}{output_path_obj.suffix}"
                ))
            
            cv2.imwrite(face_output_path, aligned_image)
            success_count += 1
        
        print(f"Processed {success_count} face(s) from '{input_path}'")
        return True

    except Exception as e:
        print(f"Error processing image '{input_path}': {str(e)}")
        return False

def get_image_paths(input_path):
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            return [str(path)]
        return []
    elif path.is_dir():
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_paths.extend([str(p) for p in path.glob(f'*{ext}')])
            image_paths.extend([str(p) for p in path.glob(f'*{ext.upper()}')])
        return image_paths
    return []

def main():
    parser = argparse.ArgumentParser(description='Align faces in images')
    parser.add_argument('input_path', type=str, 
                      help='Path to input image file or directory')
    parser.add_argument('output_path', type=str, nargs='?',
                      help='Path to output image file or directory (optional)')
    parser.add_argument('--size', type=int, default=1024, 
                      help='Output image size (default: 1024)')
    parser.add_argument('--engine', type=str, choices=['dlib', 'mediapipe'], default='mediapipe',
                      help='Face detection engine to use (default: mediapipe)')
    parser.add_argument('--debug', action='store_true',
                      help='Show facial landmarks in the output image')
    parser.add_argument('--refine', action='store_true',
                      help='Enable refined landmarks in MediaPipe (provides 478 landmarks instead of 468)')
    parser.add_argument('--max-faces', type=int, default=3,
                      help='Maximum number of faces to process per image (default: 3, use 1 for single face only)')
    args = parser.parse_args()

    # Get all image paths
    image_paths = get_image_paths(args.input_path)
    
    if not image_paths:
        print("Error: No valid images found in the provided path")
        return

    print(f"Found {len(image_paths)} images to process")
    print(f"Using {args.engine} engine for face detection")
    print(f"Processing up to {args.max_faces} face(s) per image")
    success_count = 0

    # Determine if input is a file or directory
    input_path = Path(args.input_path)
    is_file = input_path.is_file()

    # Handle output path
    if args.output_path:
        output_path = Path(args.output_path)
        if is_file and not output_path.suffix:
            output_path = output_path.with_suffix(input_path.suffix)
    else:
        if is_file:
            output_path = input_path.with_name(f"{input_path.stem}_aligned{input_path.suffix}")
        else:
            output_path = input_path / "aligned"

    # Create output directory if it doesn't exist
    if not is_file:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process files with or without progress bar
    if is_file:
        if process_image(str(input_path), str(output_path), args.size, args.engine, args.debug, args.max_faces, args.refine):
            success_count += 1
            print(f"Aligned image saved to: {output_path}")
    else:
        for image_path in tqdm(image_paths, desc="Processing images"):
            rel_path = Path(image_path).relative_to(input_path)
            out_path = str(output_path / rel_path)
            out_path = str(Path(out_path).with_name(f"{Path(out_path).stem}_aligned{Path(out_path).suffix}"))
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)

            if process_image(image_path, out_path, args.size, args.engine, args.debug, args.max_faces, args.refine):
                success_count += 1

    print(f"\nProcessing complete. Successfully processed {success_count} out of {len(image_paths)} images")

if __name__ == '__main__':
    main()
