# lightpainting
Brown CS1290 Computational Photography - Lightpainting Final Project

*Use the project by running the following commands in the project main directory:

- For light painting with green pixel tracking:
$ python main.py

- For light painting with You Only Look Once (YOLO) hand detection:
$ python main.py --method yolo 

- For light painting with the background subtraction model:
$ python main.py --method motion
\end{verbatim}

*Optional Flags:

- To specify the number of objects to track (default=1) :
$ python main.py --objects <number of objects>

- To specify the source video input (default=webcam input):
$ python main.py --source <file path to video>

- To save frames as a video in the output_videos/ directory:
$ python main.py --save

- To paint with a hard-coded color (default is rainbow):
$ python main.py --color

- To simulate motion by shifting all drawn points at each frame:
$ python main.py --shift
