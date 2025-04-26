import argparse
import os
import cv2
from face_align import FaceAlign as DlibFaceAlign
from mediapipe_face_align import FaceAlign as MediaPipeFaceAlign
from pathlib import Path

def process_image(input_path, output_size, engine='dlib'):
    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image '{input_path}'")
            return False

        if engine == 'dlib':
            face_aligner = DlibFaceAlign(output_size=output_size)
        else:  # mediapipe
            face_aligner = MediaPipeFaceAlign(output_size=output_size)

        aligned_image = face_aligner.get_aligned_image(image)

        if aligned_image is None:
            print(f"Error: No face detected in image '{input_path}'")
            return False

        base_name, ext = os.path.splitext(input_path)
        output_path = f"{base_name}_aligned{ext}"
        cv2.imwrite(output_path, aligned_image)
        print(f"Aligned image saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error processing image '{input_path}': {str(e)}")
        return False

def get_image_paths(input_paths):
    image_paths = []
    for path in input_paths:
        path = Path(path)
        if path.is_file():
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append(str(path))
        elif path.is_dir():
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.extend([str(p) for p in path.glob(f'*{ext}')])
                image_paths.extend([str(p) for p in path.glob(f'*{ext.upper()}')])
    return image_paths

def main():
    parser = argparse.ArgumentParser(description='Align faces in images')
    parser.add_argument('input_paths', nargs='+', type=str, 
                      help='Paths to input image files and/or directories')
    parser.add_argument('--size', type=int, default=1024, 
                      help='Output image size (default: 1024)')
    parser.add_argument('--engine', type=str, choices=['dlib', 'mediapipe'], default='dlib',
                      help='Face detection engine to use (default: dlib)')
    args = parser.parse_args()

    # Get all image paths
    image_paths = get_image_paths(args.input_paths)
    
    if not image_paths:
        print("Error: No valid images found in the provided paths")
        return

    print(f"Found {len(image_paths)} images to process")
    print(f"Using {args.engine} engine for face detection")
    success_count = 0
    
    for image_path in image_paths:
        if process_image(image_path, args.size, args.engine):
            success_count += 1

    print(f"\nProcessing complete. Successfully processed {success_count} out of {len(image_paths)} images")

if __name__ == '__main__':
    main()
