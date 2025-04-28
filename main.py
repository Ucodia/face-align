import argparse
import os
import cv2
from face_align import FaceAlign
from pathlib import Path
from tqdm import tqdm

def process_image(input_path, output_path, output_size, engine='dlib'):
    try:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image '{input_path}'")
            return False

        face_aligner = FaceAlign(output_size=output_size, engine=engine)
        aligned_image = face_aligner.get_aligned_image(image)

        if aligned_image is None:
            print(f"Error: No face detected in image '{input_path}'")
            return False

        cv2.imwrite(output_path, aligned_image)
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
    parser.add_argument('--engine', type=str, choices=['dlib', 'mediapipe'], default='dlib',
                      help='Face detection engine to use (default: dlib)')
    args = parser.parse_args()

    # Get all image paths
    image_paths = get_image_paths(args.input_path)
    
    if not image_paths:
        print("Error: No valid images found in the provided path")
        return

    print(f"Found {len(image_paths)} images to process")
    print(f"Using {args.engine} engine for face detection")
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
        if process_image(str(input_path), str(output_path), args.size, args.engine):
            success_count += 1
            print(f"Aligned image saved to: {output_path}")
    else:
        for image_path in tqdm(image_paths, desc="Processing images"):
            rel_path = Path(image_path).relative_to(input_path)
            out_path = str(output_path / rel_path)
            out_path = str(Path(out_path).with_name(f"{Path(out_path).stem}_aligned{Path(out_path).suffix}"))
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)

            if process_image(image_path, out_path, args.size, args.engine):
                success_count += 1

    print(f"\nProcessing complete. Successfully processed {success_count} out of {len(image_paths)} images")

if __name__ == '__main__':
    main()
