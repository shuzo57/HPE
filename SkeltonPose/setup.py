from setuptools import find_packages, setup

setup(
    name="SkeletonPose",
    version="1.0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "mmcv",
        "mmdet",
        "mmengine",
        "mmpose",
    ],
    license="MIT",
    description="A library for skeleton-based AI applications.",
)
