import cv2
import glob

fixed_face_vids = glob.glob('video_processing/video_clips_fixed_faces/*')

save_dir = 'video_processing/first_frames/'

for video_filename in fixed_face_vids:

    core_fn = video_filename.split('\\')[-1].split('.m')[0]
    print(core_fn)
    vidcap = cv2.VideoCapture(video_filename)
    success, image = vidcap.read()

    fn = save_dir + core_fn + '_first_frame.png'
    #print("writing " + fn)

    cv2.imwrite(fn, image)
