import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utilities import create_video_from_image_folder, printProgressBar

frames_folder = "trimap_frames"


def trimap(video_input):
    cap = cv2.VideoCapture(video_input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_file = "binary.avi"
    frame_num = 0
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    while cap.isOpened(): # play the video by reading frame by frame
        ret, frame = cap.read()

        printProgressBar(frame_num, total_frames)

        if (ret == True):
            frame1=frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(image, 145, 255, cv2.THRESH_BINARY_INV)
            kernel1 = np.ones((12,12), np.uint8)
            erosion = cv2.erode(thresh, kernel1, iterations=1)
            # cv2.imshow("AfterErosion", erosion)

            kernel2 = np.ones((3,3), np.uint8)
            dilation = cv2.dilate(erosion, kernel2, iterations=2)
            # cv2.imshow("AfterDilation", dilation)

            image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # plt.imshow(image)
            # plt.show()

            #sort contours
            sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

            mask = np.zeros_like(frame1)  # Create mask where white is what we want, black otherwise
            cv2.drawContours(mask, sorted_ctrs, 0, (255, 255, 255), -1)  # Draw filled contour in mask
            out = np.zeros_like(frame1)  # Extract out the object and place into output image
            out[mask == 255] = [255]

            image1 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

            # noise removal
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(image1, cv2.MORPH_OPEN, kernel, iterations=1)

            # sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=1)

            # Finding sure foreground area
            sure_fg = cv2.erode(opening, kernel, iterations=1)
            # cv2.imshow("foreground", sure_fg)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            ret, thresh = cv2.threshold(unknown, 240, 255, cv2.THRESH_BINARY)

            unknown[thresh == 255] = 127
            #cv2.imshow("unknown", unknown)

            final_mask = sure_fg + unknown
            # cv2.imshow("final_mask", final_mask)
            plt.imsave(f"{frames_folder}/trimap_mask_{frame_num+1:03d}", final_mask, cmap="gray")

            frame_num += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
        else:
            break
            if cv2.waitKey(1) == ord('q'): break


    plt.imsave(f"{frames_folder}/trimap_mask_{frame_num+1:03d}", final_mask, cmap="gray")
    cap.release()
    cv2.destroyAllWindows


    # create the video from the new trimap frames
    create_video_from_image_folder(frames_folder, output_file, shape=(1280,720))

    return frames_folder, output_file


if __name__ == "__main__":
    #test
    trimap('input.avi')
