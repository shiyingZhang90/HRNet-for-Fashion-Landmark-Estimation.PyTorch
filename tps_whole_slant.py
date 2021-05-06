import json
import numpy as np
from scipy.spatial.distance import cdist
import os

try:
    import cPickle as pickle
except ImportError:
    import pickle
import torch
import torch.nn.functional as F
import cv2
import argparse

name2id = {}
results = []


def gauss_blur(img, mask):
    '''

    :param img: HxWx3 0~255
    :param mask: HxWx1 0~1
    :return: blurred image
    '''
    band = morpho(mask, 1, True) * (1 - morpho(mask, 1, False))
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    band = band[:, :, np.newaxis]
    return img * (1 - band) + band * blur


def jpeg_blur(img, mask):
    '''
     :param img: HxWx3 0~255
    :param mask: HxWx1 0~1
    :return: degraded image
    '''
    band = morpho(mask, 1, True) * (1 - morpho(mask, 1, False))
    band = band[:, :, np.newaxis]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return img * (1 - band) + band * decimg


def morpho(mask, iter, bigger=True):
    # return mask
    mask = mask * 255
    mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # print(kernel)
    if bigger:
        mask = cv2.dilate(mask, kernel, iterations=iter)
    else:
        mask = cv2.erode(mask, kernel, iterations=iter)
    return mask / 255


def TPS(P1, P2, _lambda=1e-3, width=768, height=1024, calc_new_pos=False):
    def radius_basis(r):
        epsilon = 1e-14
        return r ** 2 * np.log(r ** 2 + epsilon)

    def homogenization(P):
        point_num = P.shape[0]
        P_homo = np.ones((point_num, 3))
        P_homo[:, 1:3] = P
        return P_homo

    point_num = P1.shape[0]
    K = radius_basis(cdist(P1, P1)) + _lambda * np.eye(point_num)
    L = np.zeros((point_num + 3, point_num + 3))
    L[:point_num, :point_num] = K
    L[:point_num, point_num:point_num + 3] = homogenization(P1)
    L[point_num:point_num + 3, :point_num] = homogenization(P1).T

    # target value, calculate in turn
    v_x = np.zeros(point_num + 3)
    v_y = np.zeros(point_num + 3)
    v_x[:point_num] = P2[:, 0]
    v_y[:point_num] = P2[:, 1]
    w_x = np.linalg.solve(L, v_x)
    a_x = w_x[point_num:]
    w_x = w_x[:point_num]
    w_y = np.linalg.solve(L, v_y)
    a_y = w_y[point_num:]
    w_y = w_y[:point_num]

    if calc_new_pos:
        points = np.zeros((width * height, 2))
        for i in range(width):
            points[i * height:(i + 1) * height, 0] = np.ones(height) * i / width
            points[i * height:(i + 1) * height, 1] = np.arange(height) / height
        h_points = homogenization(points)
        new_x = np.matmul(h_points, a_x) + np.matmul(w_x.T, radius_basis(cdist(P1, points)))
        new_y = np.matmul(h_points, a_y) + np.matmul(w_y.T, radius_basis(cdist(P1, points)))
        new_x = new_x.reshape(width, height).T
        new_y = new_y.reshape(width, height).T
        new_x = np.stack((new_x, new_y), axis=2)
    return None, new_x if calc_new_pos else None


def normalize(p, w, h):
    return p / np.array([w, h]).astype(np.float32)


def load_name_to_memory(keypoint_path):
    global results, name2id, x
    with open(keypoint_path, 'r') as f:
        results += json.load(f)
    for i in range(len(results)):
        result = results[i]
        name2id[result['image_id'].split('/')[-1]] = i
    print(name2id)


def load_keypoints(source_keypoint_path='', target_keypoint_path='',
                   w=100, h=100, source_name='', target_name=''):
    # print(source_keypoint_path, target_keypoint_path)
    if len(name2id) == 0:
        load_name_to_memory(keypoint_path=source_keypoint_path)
        load_name_to_memory(keypoint_path=target_keypoint_path)
    source_id = name2id[source_name]
    target_id = name2id[target_name]
    raw_source_keypoint = np.array(results[source_id]['keypoints'], dtype=np.float32).reshape((-1, 3))[:25, :2]
    source_keypoint = normalize(raw_source_keypoint, w, h)
    raw_target_keypoint = np.array(results[target_id]['keypoints'], dtype=np.float32).reshape((-1, 3))[:25, :2]
    target_keypoint = normalize(raw_target_keypoint, w, h)
    return source_keypoint, target_keypoint, raw_source_keypoint, raw_target_keypoint


def get_midpoint(point1, point2, x_val):
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    bias = point1[1] - slope * point1[0]
    y_val = x_val * slope + bias
    return np.array([x_val, y_val]).reshape(1, 2)


def get_slanted_x(point1, point2, shoulder, const=0.7):
    delta = point2 - point1
    if delta[1] == 0 or delta[0] == 0:
        return point2[0]
    tan_theta = delta[0] / delta[1]
    return point2[0] + tan_theta * shoulder * const


def get_align_keypoint(keypoint, is_source=True):
    if is_source:
        for i in range(11, 15):
            keypoint[i, 1] = (keypoint[i, 1] + keypoint[30 - i, 1]) / 2
            keypoint[30 - i, 1] = keypoint[i, 1]
    else:
        point1 = get_midpoint(keypoint[14, :], keypoint[16, :], keypoint[11, 0])
        point3 = get_midpoint(keypoint[14, :], keypoint[16, :], keypoint[19, 0])
        keypoint[14, :] = point1
        keypoint[16, :] = point3
        point1 = get_midpoint(keypoint[13, :], keypoint[17, :], keypoint[11, 0])
        point3 = get_midpoint(keypoint[13, :], keypoint[17, :], keypoint[19, 0])
        keypoint[13, :] = point1
        keypoint[17, :] = point3
    x = get_slanted_x(keypoint[0, :], keypoint[3, :], keypoint[5, 0] - keypoint[1, 0])
    point1 = get_midpoint(keypoint[13, :], keypoint[17, :], x)
    point2 = get_midpoint(keypoint[14, :], keypoint[16, :], x)
    point3 = get_midpoint(keypoint[13, :], keypoint[17, :], keypoint[3, 0])
    point4 = get_midpoint(keypoint[14, :], keypoint[16, :], keypoint[3, 0])

    # x = get_slanted_x(keypoint[0, :], keypoint[3, :], keypoint[5, 0] - keypoint[1, 0], const=0.9)
    # point5 = get_midpoint(keypoint[12, :], keypoint[18, :], x)
    # point6 = get_midpoint(keypoint[12, :], keypoint[18, :], keypoint[3, 0])
    align_keypoint = point2
    for i in [2, 4, 6, 11, 12, 13, 14, 16, 17, 18, 19, 24, 3, 0]:
        align_keypoint = np.concatenate((align_keypoint, keypoint[i:i + 1, :]), axis=0)
    align_keypoint = np.concatenate((align_keypoint, point4), axis=0)
    return keypoint, align_keypoint


cnt = 0


def visualize(keypoint, img_path='', output_root='./visualize_landmark', prefix='black'):
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    global cnt
    cnt += 1
    img = cv2.imread(img_path)
    for i in range(keypoint.shape[0]):
        cv2.circle(img, (int(keypoint[i, 0]), int(keypoint[i, 1])), 4, [255, 0, 170], thickness=-1)
    cv2.imwrite(os.path.join(output_root, f'{prefix}_{cnt}.jpg'), img)


def H_cosine(cloth, logo, base, name=''):
    cv2.imwrite(f'./cloth{name}.jpg', cloth)
    cv2.imwrite(f'./logo_{name}.jpg', logo)
    cloth_hsv = cv2.cvtColor(cloth, cv2.COLOR_BGR2HSV)
    logo_hsv = cv2.cvtColor(logo, cv2.COLOR_BGR2HSV)
    base_hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    cloth_h_rad = cloth_hsv[:, :, 0] / 255 * np.pi * 2
    logo_h_rad = logo_hsv[:, :, 0] / 255 * np.pi * 2
    base_h_rad = base_hsv[:, :, 0] / 255 * np.pi * 2
    return np.arccos(np.cos(cloth_h_rad - base_h_rad)), np.arccos(np.cos(logo_h_rad - base_h_rad))


def HS_cosine(cloth_hsv, logo_hsv, base_hsv, dim=0, name=''):
    if dim == 0:
        cloth_h_rad = cloth_hsv[:, :, dim] / 255 * np.pi * 2
        logo_h_rad = logo_hsv[:, :, dim] / 255 * np.pi * 2
        base_h_rad = base_hsv[:, :, dim] / 255 * np.pi * 2
        return np.arccos(np.cos(cloth_h_rad - base_h_rad)), np.arccos(np.cos(logo_h_rad - base_h_rad))
    print('base_hsv', base_hsv)
    return np.abs(cloth_hsv[:, :, dim].astype(int) - base_hsv[:, :, dim].astype(int)) / 255, np.abs(
        logo_hsv[:, :, dim].astype(int) - base_hsv[:, :, dim].astype(int)) / 255


def standardization(base, arr, mask):
    x_arr, y_arr, _ = np.nonzero(mask)
    val_arr = arr[x_arr, y_arr, :].astype(np.float32)
    mu = np.mean(val_arr, axis=0)
    scale = base[0, 0, :] / mu
    print(mu, base[0, 0, :], scale)
    arr = ((arr.astype(np.float32) - mu) * scale + base).astype(np.float32)
    return np.clip(arr, 0, 255).astype(np.uint8), base, scale, mu


def inv_standardization(arr, base, scale, mu):
    base[:, :, 0] = 0
    scale[0] = 1
    mu[0] = 0
    arr = ((arr.astype(np.float32) - base) / scale + mu).astype(np.float32)
    # x_arr, y_arr, _ = np.nonzero(mask)
    # val_arr = arr[x_arr, y_arr, :]
    # arr_mu = np.mean(val_arr, axis=0)
    # scale = mu / arr_mu
    # arr = (arr.astype(np.float32) - arr_mu) * scale + mu
    return np.clip(arr, 0, 255).astype(np.uint8)


def main(source_img_root='./data', target_img_root='./data', source_name='image_2', target_name='image_1', seg_root='', args=None,
         source_keypoint_path='', target_keypoint_path='', output_root='./output', target_folder='', target_name2='', alpha=0.8):
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    source_fn = os.path.join(source_img_root, source_name)
    target_fn = target_name

    # print(target_seg_fn)
    # source_fn = './visualize_landmark/0.jpg'
    # target_fn = './visualize_landmark/1.jpg'

    source_img = cv2.imread(source_fn)
    target_img = cv2.imread(target_fn)
    print(source_fn, target_fn)

    """
    hsv transfer color
    """
    img_hsv = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)
    mask = np.where(np.logical_and(np.logical_and(30 < img_hsv[:, :, 0], img_hsv[:, :, 0] < 77), img_hsv[:, :, 1] > 70),
                    1, 0).astype(np.uint8)
    mask = cv2.blur(cv2.blur(mask, (5, 5)), (3, 3))[:, :, np.newaxis]
    # print(mask)
    h, w, _ = target_img.shape
    x_arr, y_arr, _ = np.nonzero(mask)
    # print(x_arr, y_arr)
    x_min = max(np.min(x_arr) - 25, 0)
    y_min = max(np.min(y_arr) - 25, 0)
    x_max = min(np.max(x_arr) + 25, h - 1)
    y_max = min(np.max(y_arr) + 25, w - 1)
    crop_mask = mask[x_min:x_max, y_min:y_max, :]
    h, w, _ = crop_mask.shape
    crop_area = mask.copy()
    crop_area[x_min:x_min + h, y_min:y_min + w, :] = 1
    sh, sw, _ = source_img.shape
    if h * sw > w * sh:
        source_img = cv2.resize(source_img, (sw * h // sh, h))

    else:
        source_img = cv2.resize(source_img, (w, sh * w // sw))
    sh, sw, _ = source_img.shape
    start_h = (sh - h) // 2 + args.shift_h
    start_w = (sw - w) // 2 + args.shift_w
    pad_num = 45
    start_h += pad_num
    start_w += pad_num

    img1 = np.pad(source_img[:, :, 0], pad_num, 'symmetric')[start_h:start_h + h, start_w:start_w + w]
    img2 = np.pad(source_img[:, :, 1], pad_num, 'symmetric')[start_h:start_h + h, start_w:start_w + w]
    img3 = np.pad(source_img[:, :, 2], pad_num, 'symmetric')[start_h:start_h + h, start_w:start_w + w]

    source_img = np.concatenate([img1[:, :, np.newaxis], img2[:, :, np.newaxis], img3[:, :, np.newaxis]], 2)

    # source_img = source_img[start_h:start_h + h, start_w:start_w + w, :]
    crop_logo = source_img * crop_mask
    logo = target_img.copy()

    cv2.imwrite(f'./crop_logo_{source_name}.jpg', crop_logo)
    logo[x_min:x_min + h, y_min:y_min + w, :] = source_img

    source_fn = target_fn
    target_fn = os.path.join(target_img_root, target_name2)
    seg_fn = os.path.join(seg_root, target_name2)
    source_img = cv2.imread(source_fn)
    target_img = cv2.imread(target_fn)
    sh, sw, _ = source_img.shape
    th, tw, _ = target_img.shape
    w = max(sw, tw)
    h = max(sh, th)
    source_img = np.pad(logo, ((0, h - sh), (0, w - sw), (0, 0)), 'constant', constant_values=(255, 255))
    target_img = np.pad(target_img, ((0, h - th), (0, w - tw), (0, 0)), 'constant', constant_values=(255, 255))
    target_mask = cv2.imread(seg_fn, cv2.IMREAD_GRAYSCALE)
    target_mask = np.pad(target_mask, ((0, h - th), (0, w - tw)), 'constant', constant_values=(0, 0))
    cv2.imwrite(f'./source_{source_name}.jpg', source_img)
    cv2.imwrite(f'./target_{source_name}.jpg', target_img)
    source_keypoint, target_keypoint, raw_source_keypoint, raw_target_keypoint = \
        load_keypoints(w=w, h=h, source_name=target_name.split('/')[-1], target_name=target_name2,
                       source_keypoint_path=source_keypoint_path, target_keypoint_path=target_keypoint_path)
    raw_target_keypoint, target_keypoint = get_align_keypoint(raw_target_keypoint, is_source=False)
    raw_source_keypoint, source_keypoint = get_align_keypoint(raw_source_keypoint, is_source=True)
    visualize(target_keypoint, target_fn)
    visualize(source_keypoint, source_fn)
    target_keypoint = normalize(target_keypoint[:-2, :], w, h)
    source_keypoint = normalize(source_keypoint[:-2, :], w, h)
    _, grid = TPS(target_keypoint, source_keypoint, width=w, height=h, _lambda=args._lambda,
                  calc_new_pos=True)

    grid = torch.from_numpy(grid)
    # 619 246
    # tensor([0.2597, 0.6458], dtype=torch.float64)
    source_img = torch.from_numpy(source_img.astype(np.float64)).unsqueeze(dim=0).permute(0, 3, 1, 2)
    target_img = torch.from_numpy(target_img.astype(np.float64)).unsqueeze(dim=0).permute(0, 3, 1, 2)
    # print(grid)
    grid = grid.unsqueeze(dim=0) * 2 - 1.0
    # print(grid.shape)
    # print(grid)
    warp_img = F.grid_sample(source_img, grid, mode='bilinear', padding_mode='border')
    warp_img = warp_img.squeeze(dim=0).permute(1, 2, 0)
    warp_img = warp_img.numpy().astype(np.uint8)
    target_img = target_img.squeeze(dim=0).permute(1, 2, 0)
    target_img = target_img.numpy().astype(np.uint8)
    img_hsv = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)

    # mask = np.where(np.logical_and(np.logical_and(30 < img_hsv[:, :, 0], img_hsv[:, :, 0] < 77), img_hsv[:, :, 1] > 70),
    #                 1, 0).astype(np.uint8)
    # mask = cv2.blur(cv2.blur(mask, (5, 5)), (3, 3))[:, :, np.newaxis]
    # hsv_base = cv2.cvtColor(np.array([230, 230, 230], dtype=np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)
    # new_img_hsv, base, scale, mu = standardization(hsv_base, target_img, mask)
    # target_img = cv2.cvtColor(new_img_hsv, cv2.COLOR_HSV2BGR) * mask + target_img * (1 - mask)
    # name = '.'.join((target_name2.split('/')[-1]).split('.')[:-1])
    # cv2.imwrite(f'./{name}.jpg', target_img)
    # cv2.imwrite(f'./{name}_mask.jpg', mask * 255)
    mask = target_mask.astype(float) / 255

    mask = mask[:, :, np.newaxis]
    warp_img = warp_img.astype(np.float32) * target_img.astype(np.float32) / 255
    warp_img = warp_img.astype(np.float32) * alpha + target_img.astype(np.float32) * (1 - alpha)

    cv2.imwrite('./warp.jpg', warp_img)

    warp_img = (mask * warp_img + (1 - mask) * target_img).astype(np.uint8)

    warp_img = gauss_blur(warp_img, mask)
    warp_img = jpeg_blur(warp_img, mask)

    result = warp_img  # (mask * warp_img + (1 - mask) * target_img).astype(np.uint8)

    cv2.imwrite(os.path.join(output_root, '.'.join((source_name.split('/')[-1]).split('.')[:-1]) + '_' + '.'.join(
        (target_name2.split('/')[-1]).split('.')[:-1]) + '.jpg'), result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logo_root', type=str, default='root', help='name of the dataset.')
    parser.add_argument('--model_root', type=str, default='root', help='name of the dataset.')
    parser.add_argument('--logo_keypoint_root', type=str,
                        default='./output/deepfashion2/pose_hrnet/w48_384x288_adam_lr1e-3/logo/results/keypoints_test_results_0.json',
                        help='name of the dataset.')
    parser.add_argument('--model_keypoint_root', type=str,
                        default='./output/deepfashion2/pose_hrnet/w48_384x288_adam_lr1e-3/model/results/keypoints_test_results_0.json',
                        help='name of the dataset.')
    parser.add_argument('--output_root', type=str, default='./results', help='name of the dataset.')
    parser.add_argument('--seg_root', type=str, default='')
    parser.add_argument('--template_path', type=str, default='./template2/p1/Lark20210505-144716.png',
                        help='name of the dataset.')
    parser.add_argument('--shift_h', type=int, default=0)
    parser.add_argument('--shift_w', type=int, default=0)
    parser.add_argument('--_lambda', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=0.6)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)
    for source in os.listdir(args.logo_root):
        for target_folder in os.listdir(args.model_root):
            # print(source, target)
            output_root = os.path.join(args.output_root, source.split('.')[0] + '_' + target_folder)
            target_root = os.path.join(args.model_root, target_folder)
            seg_root = os.path.join(args.seg_root, target_folder)

            for target in os.listdir(target_root):
                print(source, target_folder, target)
                main(source_img_root=args.logo_root, target_img_root=target_root, target_name=args.template_path,
                     seg_root=seg_root, args=args, alpha=args.alpha,
                     source_keypoint_path=args.logo_keypoint_root, target_keypoint_path=args.model_keypoint_root,
                     output_root=output_root, source_name=source, target_name2=target, target_folder=target_folder)
