import cv2
from tracking_functions import *
from utilities import printProgressBar


def track(image_dir_path, s_initial):

    image_name = "trackedFrame_"

    # SET NUMBER OF PARTICLES
    N = 100

    # CREATE INITIAL PARTICLE MATRIX 'S' (SIZE 6xN)
    S_temp = (s_initial) * np.ones((N, 1))
    S = predict_particles(np.transpose(S_temp))

    # LOAD FIRST IMAGE
    I = cv2.imread(image_dir_path + os.sep + "frame_001.jpg")

    # COMPUTE NORMALIZED HISTOGRAM
    q = comp_norm_hist(I, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W = np.zeros(shape=(N, 1))
    C = np.zeros(shape=(N, 1))

    for i in range(N):
        W[i] = comp_bat_dist(comp_norm_hist(I, S[:, i]), q)

    W = W / np.sum(W)
    C[0] = W[0]
    for i in range(1, N):
        C[i] = C[i - 1] + W[i]

    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(image_dir_path)
    length = len(image_name_list)

    for index, image_name in enumerate(image_name_list[1:]):

        printProgressBar(index, length)

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = image_dir_path + os.sep + image_name
        I = cv2.imread(image_path)

        # plt.imshow(I)
        # plt.show()

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE)
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)

        W = np.zeros(shape=(N, 1))
        C = np.zeros(shape=(N, 1))

        for k in range(N):
            W[k] = comp_bat_dist(comp_norm_hist(I, S[:, k]), q)

        W = W / np.sum(W)
        C[0] = W[0]
        for k in range(1, N):
            C[k] = C[k - 1] + W[k]

        # CREATE DETECTOR PLOTS
        images_processed += 1
        show_particles(I, S, W, images_processed, image_name)

    comp_norm_hist(I, s_initial)


if __name__ == "__main__":
    pass
