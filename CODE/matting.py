import cv2
import numpy as np
import matplotlib.pyplot as plt
import utilities as ut
import os


def add_background_image(trimap_folder, input_vid_path, background_image_path):
    im_shape = (720, 1280, 3)
    output_file_path = "matted.avi"
    input_cap = cv2.VideoCapture(input_vid_path)
    bg_image = cv2.imread(background_image_path)
    bg_image = cv2.resize(bg_image, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)

    alpha = 0.75

    # plt.imshow(bg_image)
    # plt.show()

    fouorcc = cv2.VideoWriter_fourcc(*"MJPG")

    image_name_list = os.listdir(trimap_folder)
    length = len(image_name_list)

    input_ret, input_frame = input_cap.read()
    vid_writer = cv2.VideoWriter(output_file_path, fouorcc, 30, (input_frame.shape[1], input_frame.shape[0]))

    frame_num = 0
    total_frames = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while input_cap.isOpened():
        trimap_frame = cv2.imread(trimap_folder + '/' + image_name_list[frame_num])

        if (input_ret):

            ut.printProgressBar(frame_num + 1, total_frames)

            # initialize new image to be written
            out_frame = np.zeros(im_shape, dtype=np.uint8)

            # iterate over all pixels
            for row in range(out_frame.shape[0]):
                for col in range(out_frame.shape[1]):
                    if trimap_frame[row,col].max() > 200:
                        out_frame[row,col] = input_frame[row,col]
                    elif 100 < np.average(trimap_frame[row,col]) < 150:
                        out_frame[row,col] = alpha * bg_image[row,col] + (1-alpha) * input_frame[row,col]
                    else:
                        out_frame[row,col] = bg_image[row,col]
                    # print("d")
            #     check if binary mask value is 0, if it is take the value from the background image,
            #     otherwise from the frame

            # cv2.imshow('BGSUB', out_frame)
            # plt.imshow(out_frame)
            # plt.show()
            vid_writer.write(out_frame)
            # plt.imshow(out_frame)
            # plt.show()
            frame_num += 1

            input_ret, input_frame = input_cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
        else:
            break

    vid_writer.release()
    input_cap.release()
    cv2.destroyAllWindows()

    return output_file_path

def cmb(fg,bg,a):
    return fg * a + bg * (1-a)

def main():
    foreground = cv2.VideoCapture('output.avi')
    background = cv2.imread('background2.jpg')
    alpha = cv2.VideoCapture('circle_alpha.mp4')

    while foreground.isOpened():
        fr_foreground = foreground.read()[1]/255
        fr_background = background/255
        fr_alpha = 0.75

        cv2.imshow('My Image',cmb(fr_foreground,fr_background,fr_alpha))

        if cv2.waitKey(1) == ord('q'): break

    cv2.destroyAllWindows

if __name__ == '__main__':
    main()