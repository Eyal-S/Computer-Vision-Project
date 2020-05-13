import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as ptc
import cv2


def comp_norm_hist(I, s):
    x_range_max = np.maximum(s[1] - s[3], 1)
    x_range_min = np.minimum(s[1] + s[3], I.shape[0])
    y_range_max = np.maximum(s[0] - s[2], 1)
    y_range_min = np.minimum(s[0] + s[2], I.shape[1])

    x_range = np.array(range(int(x_range_max), int(x_range_min)))
    y_range = np.array(range(int(y_range_max), int(y_range_min))).reshape(-1, 1)

    I_object = I[x_range, y_range, :].copy()
    I_object = np.floor((np.double(I_object)) / 16)
    hist = np.zeros((4096, 1))

    for m in range(I_object.shape[0]):
        for n in range(I_object.shape[1]):
            index = I_object[m, n, 0] + 16 * I_object[m, n, 1] + 256 * I_object[m, n, 2]
            hist[int(index)] = hist[int(index)] + 1

    norm_hist = hist / (np.sum(hist))
    return norm_hist


def comp_bat_dist(p, q):
    w = np.exp((20 * np.sum(np.sqrt(p * q))))
    return w


def predict_particles(S_next_tag):
    variances = [1, 1, 0, 0, 1, 1]

    S_next = S_next_tag.copy()
    S_next[0, :] = S_next_tag[0, :].copy() + S_next_tag[4].copy()
    S_next[1, :] = S_next_tag[1, :].copy() + S_next_tag[5].copy()

    for i in range(S_next_tag.shape[1]):
        Noise = (np.round(3 * np.random.standard_normal((6,))) * np.transpose(variances))
        S_next[:, i] = S_next[:, i] + Noise

    return S_next


def sample_particles(S_prev, C):
    S_next_tag = np.zeros(S_prev.shape)
    for i in range(S_prev.shape[1]):
        r = np.random.rand()
        j = 0
        while (C[j] < r) & (j < 99):
            j = j + 1

        S_next_tag[:, i] = S_prev[:, j]

    return S_next_tag


def show_particles(I, S, W, i, ID):
    title = f"{ID} - Frame number = {i:03d}"
    title2 = f"{ID}-{i:03d}"
    output_dir = "tracked_images"

    # create output directory if doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax1 = plt.subplots(1)
    fig.suptitle(title)

    x_width = S[4, 1]
    y_width = S[3, 1]
    x_average = np.floor(np.mean(S[0, :])) - x_width
    y_average = np.floor(np.mean(S[1, :])) - y_width
    I_max = np.argmax(W)

    # Create a Rectangle patch
    rect1 = ptc.Rectangle((S[0, I_max] - x_width, S[1, I_max] - y_width), 2 * x_width, 2 * y_width, linewidth=1,
                          edgecolor='r', facecolor='none')
    rect2 = ptc.Rectangle((x_average, y_average), 2 * x_width, 2 * y_width, linewidth=1, edgecolor='g',
                          facecolor='none')

    # Add the patch to the Axes
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)

    # Display the image
    ax1.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    # plt.show()

    # saving the figure
    # print(f"saving image {title2}")
    fig.savefig(fname=f"{output_dir}/{title2}.png", bbox_inches='tight')

    plt.close(fig)
