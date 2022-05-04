import json
import cv2
import os
import glob

frame_skip = 5
max_frames = 2000
video_dir = 'video_processing/video_clips_fixed_faces/'
video_files = glob.glob(video_dir + '*.mp4')
annotation_dir = 'video_processing/first_frames/'

save_dir = 'video_processing/face_frames/'

for vf in video_files[:2]:
    core_vf = vf.split('\\')[-1].split('.mp4')[0]
    # could replace spaces with underscores...
    annotation_fn = annotation_dir + core_vf + '_first_frame.json'

    print("loading video: ", core_vf)
    vidcap = cv2.VideoCapture(vf)

    print("loading annotations", annotation_fn)
    with open(annotation_fn, 'r') as f:
        data = json.load(f)
    people_annotations = data['shapes']

    print('creating a folder for each annotation')
    for i in range(len(people_annotations)):
        newpath = save_dir + core_vf + "_" + str(i)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    success = True
    frame_num = 0
    while success and frame_num < max_frames:
        frame_num += 1

        if frame_num % 100 == 0:
            print('frame: ', frame_num)

        for _ in range(frame_skip):
            if success: #prevents reading from closed stream before outerloop
                success, image = vidcap.read()

        for person_num, box in enumerate(people_annotations):
            TL = box['points'][0]
            BR = box['points'][1]

            TL = (int(TL[0]), int(TL[1]))
            BR = (int(BR[0]), int(BR[1]))

            try:

                copy_region = image[TL[1]:BR[1],TL[0]:BR[0]]
                copy_region_resize = cv2.resize(copy_region, (256,256))

                save_fn = save_dir + core_vf + "_" + str(person_num) + '/' + f'{frame_num:06}' + '.png'

                cv2.imwrite(save_fn, copy_region_resize)
            except:
                continue

    print('------------------------------------')
