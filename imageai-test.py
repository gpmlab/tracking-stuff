from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

detector.CustomObjects(sports_ball=True)

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "foos.mov"),
                                output_file_path=os.path.join(execution_path, "ball_detected")
                                , frames_per_second=20, log_progress=True)