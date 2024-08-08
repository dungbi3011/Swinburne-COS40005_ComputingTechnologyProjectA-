import cv2

import subprocess

import os

unconverted_data_directory = 'unconverted_data/Normal_Crowds'

training_data_directory = 'training_data'

cwd = os.getcwd()

extracted_frames_folder = cwd + "/extracted_frames"

def convert_data(directory):
    # run terminal command to convert avi video into mp4 video
    for filename in os.listdir(directory):
        subprocess.run(['ffmpeg',
                        '-i',
                        directory + '/' + filename,
                        '-qscale',
                        '0',
                        training_data_directory + '/' + filename.split('.')[0] + '.mp4',
                        '-loglevel',
                        'quiet'])


def load_video(directory):

    for filename in os.listdir(directory):

        os.mkdir(extracted_frames_folder + '/' + filename.split('.')[0])

        current_video_extracted_frames_folder = extracted_frames_folder + '/' + filename.split('.')[0]

        cap = cv2.VideoCapture(directory + '/' + filename)

        frame_rate = round(cap.get(cv2.CAP_PROP_FPS))

        FRAME_SAVE_PER_SECOND = 32
        frame_save_counter = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success: 
                break

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
            if frame_save_counter < FRAME_SAVE_PER_SECOND:
                file_name = current_video_extracted_frames_folder + "/frame_" + str(frame_id) + ".jpg"
                # file_name = extracted_frames_folder + "/frame_" + str(frame_id) + ".jpg"
                cv2.imwrite(file_name, frame)
                frame_save_counter += 1

            if (frame_id % frame_rate == 0): 
                frame_save_counter = 0

        cap.release()

convert_data(unconverted_data_directory)
load_video(training_data_directory)





