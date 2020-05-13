import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_pyramid(image, num_levels):
    pyramid = [None for _ in range(num_levels)]
    pyramid[0] = image
    for i in range(1, num_levels):
        pyramid[i] = cv2.pyrDown(pyramid[i - 1], pyramid[i])
    return pyramid


def lucas_kanade_step(I1, I2, WindowSize):
    (M, N) = I2.shape
    du = np.zeros(shape=(M, N))
    dv = np.zeros(shape=(M, N))

    I_x_img = (I1 - np.roll(I1, axis=1, shift=1))
    I_y_img = (I1 - np.roll(I1, axis=0, shift=1))
    I_t_img = I2 - I1

    I_x_img[:, :1] = 0
    I_y_img[:1, :] = 0

    window_width = WindowSize // 2

    for row in range(1, M - 1):
        for col in range(1, N - 1):
            i = np.arange(max(row - window_width, 0), min(row + window_width, M))
            j = np.arange(max(col - window_width, 0), min(col + window_width, N))
            I_x = np.zeros(shape=(len(i), len(j)))
            I_y = np.zeros(shape=(len(i), len(j)))
            I_t = np.zeros(shape=(len(i), len(j)))
            for r in range(len(i)):
                for c in range(len(j)):
                    I_x[r, c] = 0.5 * I_x_img[r, c]
                    I_y[r, c] = 0.5 * I_y_img[r, c]
                    I_t[r, c] = 0.5 * I_t_img[r, c]
            I_x = np.reshape(I_x, newshape=(I_x.size, 1))
            I_y = np.reshape(I_y, newshape=(I_y.size, 1))

            B = np.concatenate((I_x, I_y), axis=1)
            B_trans_B = np.matmul(B.T, B)

            I_t = np.reshape(I_t, newshape=(I_t.size, 1))

            delta_p = -np.matmul(np.linalg.pinv(B_trans_B), np.matmul(B.T, I_t))

            du[row, col] = delta_p[0]
            dv[row, col] = delta_p[1]

    return [du, dv]


def warp_image(I2, u, v):
    flow = np.ndarray(shape=(u.shape[0], u.shape[1], 2))
    flow[:, :, 0] = u
    flow[:, :, 1] = v

    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]

    flow = np.ndarray.astype(flow, dtype=np.float32)
    res = cv2.remap(I2, flow, None, cv2.INTER_LINEAR)

    return res


def lucas_kanade_optical_flow(I1, I2, WindowSize, MaxIter, NumLevels):
    # build pyramids
    pyramid1 = build_pyramid(I1, NumLevels)
    pyramid2 = build_pyramid(I2, NumLevels)

    u = np.zeros(shape=I2.shape)
    v = np.zeros(shape=I2.shape)

    # iterate over levels of the pyramids
    for level in range(NumLevels - 1, -1, -1):
        M, N = pyramid1[level].shape

        u = cv2.resize(u, dsize=(N, M), interpolation=cv2.INTER_LINEAR)
        u *= 2
        v = cv2.resize(v, dsize=(N, M), interpolation=cv2.INTER_LINEAR)
        v *= 2

        temp_image = warp_image(pyramid2[level], u, v)

        for iter in range(MaxIter):
            # print(f"Starting level {level}, iteration {int(iter+1)}")
            du, dv = lucas_kanade_step(pyramid1[level], temp_image, WindowSize)
            u += du
            v += dv
            temp_image = warp_image(temp_image, du, dv)

            # fig = plt.figure(figsize=(2,1))
            # fig.add_subplot(1,2, 1)
            # plt.imshow(pyramid1[level])
            # fig.add_subplot(1,2, 2)
            # plt.imshow(temp_image)

            # plt.show()
    return u, v


def lucas_kanade_video_stabilization(InputVid, WindowSize, MaxIter, NumLevels, NumOfFrames):
    outputfileName = 'StabilizedVid.avi'  # change the file name if needed
    frame_per_second = 30.0

    # Read input video
    cap = cv2.VideoCapture(InputVid)

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up output video
    writer = cv2.VideoWriter(outputfileName, cv2.VideoWriter_fourcc(*"MJPG"), frame_per_second, (w, h), 0)

    # Read first frame
    _, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    (M, N) = np.shape(prev_gray)
    u = np.zeros(shape=(M, N))
    v = np.zeros(shape=(M, N))
    FrameNum = 1

    while (cap.isOpened() and FrameNum < NumOfFrames):  # play the video by reading frame by frame
        print(f"processing frame {FrameNum} / {NumOfFrames}")
        ret, frame = cap.read()
        if ret == True:
            CurrFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            (du, dv) = lucas_kanade_optical_flow(prev_gray, CurrFrame, WindowSize, MaxIter, NumLevels)
            u = u + du
            v = v + dv
            WarpFrame = warp_image(CurrFrame, u, v)
            writer.write(WarpFrame)  # save the frame into video file
            prev_gray = CurrFrame
            FrameNum = FrameNum + 1

            # Display the resulting image
            cv2.imshow('StabilizedVid', WarpFrame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break
        else:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    return outputfileName


if __name__ == "__main__":
    pass
