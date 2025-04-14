import argparse
import os
import cv2
from face_align import FaceAlign

def main():
    parser = argparse.ArgumentParser(description='Align faces in images')
    parser.add_argument('input_image', type=str, help='Path to input image file')
    parser.add_argument('--size', type=int, default=1024, help='Output image size (default: 1024)')
    args = parser.parse_args()

    if not os.path.exists(args.input_image):
        print(f"Error: Input file '{args.input_image}' does not exist")
        return

    base_name, ext = os.path.splitext(args.input_image)
    output_path = f"{base_name}_aligned{ext}"

    try:
        image = cv2.imread(args.input_image)
        if image is None:
            print(f"Error: Could not read image '{args.input_image}'")
            return

        face_aligner = FaceAlign(output_size=args.size)
        aligned_image = face_aligner.get_aligned_image(image)

        if aligned_image is None:
            print("Error: No face detected in the image")
            return

        cv2.imwrite(output_path, aligned_image)
        print(f"Aligned image saved to: {output_path}")

    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == '__main__':
    main()
