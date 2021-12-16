import numpy as np


def pull_next_level(image, weights):
    w = np.shape(image)[0]
    hw = w // 2

    im_next, weights_next = np.zeros((hw, hw, np.shape(image)[2])), np.zeros((hw, hw))

    for j in range(hw):
        for i in range(hw):
            y, x = 2 * j, 2 * i

            r_n = []
            g_n = []
            b_n = []

            weights_next[j, i] = max([weights[y, x], weights[y + 1, x], weights[y + 1, x + 1], weights[y, x + 1]])

            if (weights[y, x] == 0).all():
                pass
            else:
                r_n.append(image[y, x, 0])
                g_n.append(image[y, x, 1])
                b_n.append(image[y, x, 2])

            if (weights[y + 1, x] == 0).all():
                pass
            else:
                r_n.append(image[y + 1, x, 0])
                g_n.append(image[y + 1, x, 1])
                b_n.append(image[y + 1, x, 2])

            if (weights[y + 1, x + 1] == 0).all():
                pass
            else:
                r_n.append(image[y + 1, x + 1, 0])
                g_n.append(image[y + 1, x + 1, 1])
                b_n.append(image[y + 1, x + 1, 2])

            if (weights[y, x + 1] == 0).all():
                pass
            else:
                r_n.append(image[y, x + 1, 0])
                g_n.append(image[y, x + 1, 1])
                b_n.append(image[y, x + 1, 2])

            r = np.mean(r_n)
            g = np.mean(g_n)
            b = np.mean(b_n)

            im_next[j, i, 0], im_next[j, i, 1], im_next[j, i, 2] = r, g, b

    return im_next, weights_next


def build_pyramid(image, weights):
    pyramid_im, pyramid_w = [image], [weights]

    running = True
    while running:
        if np.shape(image)[0] < 2:
            running = False
        else:
            image, weights = pull_next_level(image, weights)
            pyramid_im.append(image)
            pyramid_w.append(weights)

    return pyramid_im, pyramid_w


def project_point_up(px_coordinate_in_down):
    y, x = px_coordinate_in_down
    d_x, d_y = 0.25, 0.25

    running = True
    while running:
        if y % 2 == 0:
            y -= 1
            d_y += 0.5
        else:
            running = False

    running = True
    while running:
        if x % 2 == 0:
            x -= 1
            d_x += 0.5
        else:
            running = False

    return np.array([y // 2, x // 2]), np.array([d_y, d_x])


def bilinear_interpolation(col_a, col_b, col_c, col_d, x):
    alpha, beta = x
    f_0 = col_d * (1- alpha) + col_c * alpha
    f_1 = col_a * (1 - alpha) + col_b * alpha
    f_x = f_0 * (1 - beta) + f_1 * beta  # not sure if beta or 1 - beta
    return f_x


def push_pyramid(im_pyramid, w_pyramid):
    im_pyramid.reverse()
    w_pyramid.reverse()

    layer = 0
    for pyramid in w_pyramid:
        column_pos = 0
        for column in pyramid:
            for entry in range(len(column)):
                if column[entry] == 0:
                    index_start, subpixel_position = project_point_up([column_pos, entry])

                    a_idx = [index_start[0], index_start[1]]
                    b_idx = [index_start[0], index_start[1] + 1]
                    c_idx = [index_start[0] + 1, index_start[1] + 1]
                    d_idx = [index_start[0] + 1, index_start[1]]
                    x = subpixel_position

                    col_a = im_pyramid[layer - 1][a_idx[0]][a_idx[1]]
                    col_b = im_pyramid[layer - 1][b_idx[0]][b_idx[1]]
                    col_c = im_pyramid[layer - 1][c_idx[0]][c_idx[1]]
                    col_d = im_pyramid[layer - 1][d_idx[0]][d_idx[1]]

                    im_pyramid[layer][column_pos][entry] = bilinear_interpolation(col_a, col_b, col_c, col_d, x)

            column_pos += 1

        layer += 1

    return im_pyramid, w_pyramid


def remove_text(image):
    # only pull non red pixels
    width, height = image.shape[0], image.shape[1]
    weights = np.ones((width, height))

    column_pos = 0
    for column in image:
        for entry in range(column.shape[0]):
            if (column[entry] == np.array([1, 0, 0])).all():
                weights[column_pos][entry] = 0

        column_pos += 1

    pyramid_im, pyramid_w = build_pyramid(image, weights)

    im_pyramid, w_pyramid = push_pyramid(list(pyramid_im), list(pyramid_w))

    return np.array(im_pyramid[-1])
