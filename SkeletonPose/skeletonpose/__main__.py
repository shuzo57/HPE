import argparse
import os
import sys

from .run import Run


def main():
    parser = argparse.ArgumentParser(
        description="Run pose estimation and save the results."
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-pc",
        "--pose_config",
        required=True,
        help="Path to the pose configuration file.",
    )
    parser.add_argument(
        "-pck",
        "--pose_checkpoint",
        required=True,
        help="Path to the pose checkpoint file.",
    )
    parser.add_argument(
        "-dc",
        "--det_config",
        required=True,
        help="Path to the detection configuration file.",
    )
    parser.add_argument(
        "-dck",
        "--det_checkpoint",
        required=True,
        help="Path to the detection checkpoint file.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        sys.exit(f"Input file does not exist: {args.input_path}")

    Run(
        input_path=args.input_path,
        output_path=args.output_path,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_checkpoint,
        det_config=args.det_config,
        det_checkpoint=args.det_checkpoint,
    )


if __name__ == "__main__":
    main()
