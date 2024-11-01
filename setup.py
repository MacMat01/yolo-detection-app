from setuptools import setup, find_packages

setup(name='yolo-detection-app', version='1.0.0', license='MIT',
      description='A project for detecting cards using YOLOv8', author='MacMat01, SenseiBonsai2k',
      author_email='matteo.machella01@gmail.com, cristian.marinozzi1@gmail.com',
      url='https://github.com/MacMat01/yolo-detection-app', packages=find_packages(),
      install_requires=['ultralytics', 'opencv-python-headless',  # py-opencv
                        'pyzbar'], classifiers=['Development Status :: 3 - Alpha', 'Intended Audience :: Developers',
                                                'License :: OSI Approved :: MIT License',
                                                'Programming Language :: Python :: 3.12.3', ], )
