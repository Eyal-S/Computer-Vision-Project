import stabilization as st
import matting as mt
import tracking as tr
import utilities as ut
import trimap as tri
import winsound


def runme(input_vid_path, background_image_path, s_initial):

    # 1 - stabilize the input vid
    stabilized_vid = st.stabilize(input_vid_path)

    # 2 - subtract background from stabilized video
    print("Starting trimap")
    trimap_folder, trimap_vid = tri.trimap(stabilized_vid)
    print("Done with trimap")

    # 3 - apply matting
    print("Starting matting")
    matted_vid = mt.add_background_image(trimap_folder, stabilized_vid,
                                         background_image_path)
    print("Done with matting")

    # generate frames folder
    ut.save_video_frames(matted_vid, folder_name="matted_vid_4_frames")

    # 4 - apply tracking
    print("Starting tracking")
    output_vid = tr.track("matted_vid_4_frames", s_initial)

    # 5 - generate the output video from the tracked frames
    ut.create_video_from_image_folder("tracked_images", "OUTPUT.avi", shape=(555,416))
    print("Done tracking")

    return output_vid


if __name__ == "__main__":

    # print("starting")
    input_vid_path = "../Input/INPUT.avi"
    background_image_path = "../Input/background.jpg"

    # Initial Settings
    s_initial = [920,   # x center
                 300,   # y center
                 20,    # half width
                 100,   # half height
                 -5,    # velocity x
                 0]     # velocity y

    runme(input_vid_path, background_image_path, s_initial)

    # # This is used to notify when the code finishes
    # winsound.Beep(1760, 1000)
