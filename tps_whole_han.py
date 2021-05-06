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
        # load_name_to_memory(keypoint_path=source_keypoint_path)
        load_name_to_memory(keypoint_path=target_keypoint_path)
    # source_id = name2id[source_name]
    target_id = name2id[target_name]
    # raw_source_keypoint = np.array(results[source_id]['keypoints'], dtype=np.float32).reshape((-1, 3))[:25, :2]
    # source_keypoint = normalize(raw_source_keypoint, w, h)
    raw_target_keypoint = np.array(results[target_id]['keypoints'], dtype=np.float32).reshape((-1, 3))[:25, :2]
    target_keypoint = normalize(raw_target_keypoint, w, h)
    return  target_keypoint, raw_target_keypoint


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


def main(source_img_root='./data', target_img_root='./data', source_name='image_2', target_name='image_1',
         source_keypoint_path='', target_keypoint_path='', output_root='./output', target_folder=''):
    print(target_name)
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    source_fn = os.path.join(source_img_root, source_name)
    target_fn = os.path.join(target_img_root, target_name)
    cloth_mask_fn=os.path.join(target_img_root.replace('model','mask'),target_name)

    target_seg_fn = os.path.join('./segmentation/segmentation_model/gray_atr/' + target_folder,
                                 '.'.join(target_name.split('.')[:-1]) + '.png')
    # print(target_seg_fn)
    # source_fn = './visualize_landmark/0.jpg'
    # target_fn = './visualize_landmark/1.jpg'
    source_img = cv2.imread(source_fn)
    target_img = cv2.imread(target_fn)
    cloth_mask= cv2.imread(cloth_mask_fn,0)/255.0 ## cloth mask

    target_seg = cv2.imread(target_seg_fn, 0)
    print(type(target_seg))
    #target_seg = (target_seg == 4).astype(np.float64)
    sh, sw, _ = source_img.shape
    th, tw, _ = target_img.shape
    # w = max(sw, tw)
    # h = max(sh, th)
    # target_seg = np.pad(target_seg, ((0, h - th), (0, w - tw)), 'constant', constant_values=(0, 0))
    # target_seg = np.expand_dims(target_seg, axis=2)
    # source_img = np.pad(source_img, ((0, h - sh), (0, w - sw), (0, 0)), 'constant', constant_values=(255, 255))
    # target_img = np.pad(target_img, ((0, h - th), (0, w - tw), (0, 0)), 'constant', constant_values=(255, 255))
    #


    target_keypoint, raw_target_keypoint = \
        load_keypoints(w=tw, h=th, source_name=source_name, target_name=target_name,
                       source_keypoint_path=source_keypoint_path, target_keypoint_path=target_keypoint_path)
    # raw_target_keypoint, target_keypoint = get_align_keypoint(raw_target_keypoint, is_source=False)
    # raw_source_keypoint, source_keypoint = get_align_keypoint(raw_source_keypoint, is_source=True)

    # print('---------------------------------------------------------------------------')
    # print(raw_target_keypoint.shape)
    """
    hsv transfer color
    """

    img_hsv = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)

    # Here extracts the cloth mask from green cloth comment
    # mask = np.where(np.logical_and(np.logical_and(30 < img_hsv[:, :, 0], img_hsv[:, :, 0] < 77), img_hsv[:, :, 1] > 70),1, 0).astype(np.uint8)

    #print(cloth_mask.shape)
    mask=cloth_mask # with manual mask
    mask*=(cloth_mask>0.1).astype(np.float32)

    mask = cv2.blur(cv2.blur(mask, (1, 1)), (1, 1))[:, :, np.newaxis]
    new_img_hsv = img_hsv.copy()
    #comment for green cloth
    # cloth = mask * new_img_hsv


    hsv_base = cv2.cvtColor(np.array([230, 230, 230], dtype=np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)
    print('hsv_base', hsv_base)


    # new_img_hsv, base, scale, mu = standardization(hsv_base, cloth, mask)


    ## Here transforms green to white so we comment this line
    # target_img = cv2.cvtColor(new_img_hsv, cv2.COLOR_HSV2BGR) * mask + target_img * (1 - mask)



    visualize(raw_target_keypoint, img_path=target_fn, output_root='./visualize_landmark', prefix='black')

    cv2.imwrite(f'./target_transfer_{target_name}_mask.jpg', mask * 255)
    cv2.imwrite(f'./target_transfer_{target_name}.jpg', target_img)
    h, w, _ = target_img.shape
    h_target,w_target,_=target_img.shape
    ## get the bounding box of the mask
    x_arr, y_arr, _ = np.nonzero(mask)
    x_min = max(np.min(x_arr) - 10, 0)
    y_min = max(np.min(y_arr) - 10, 0)
    x_max = min(np.max(x_arr) + 10, h - 1)
    y_max = min(np.max(y_arr) + 10, w - 1)
    crop_mask = mask[x_min:x_max, y_min:y_max, :] ## remain consistency
    h, w, _ = crop_mask.shape
    crop_area = mask.copy()
    crop_area[x_min:x_min+h, y_min:y_min+w, :] = 1
    sh, sw, _ = source_img.shape ## get the texture shape
    scale = max(h / sh, w / sw)
    if h * sw > w * sh:
        source_img = cv2.resize(source_img, (sw * h // sh, h))
    else:
        source_img = cv2.resize(source_img, (w, sh * w // sw))
    sh, sw, _ = source_img.shape
    point1=raw_target_keypoint[7-1]
    point2=raw_target_keypoint[25-1]
    mid_point=(point1+point2)/2
    print('midpoint-----------------------')
    print(mid_point[0]/sw)
    mid_x=raw_target_keypoint[4-1,0]
    print ('mid_collar ', mid_x/sw)
    start_h = (sh - h) // 2   #x,y start location
    start_w = (sw - w) // 2     # print (source_img.shape)
    


    print (start_h,start_w)

    pad_num=45
    start_h+=pad_num
    start_w+=pad_num

    correct_ratio=1

    delta_w=int(((x_min+x_max)/2-mid_x)*correct_ratio)
    #start_w -=delta_w
    print("start_w ", start_w, "delta_w ", delta_w)
    if (start_w > delta_w and delta_w>60):
        start_w += 0
    elif (start_w > delta_w and delta_w<0):
        start_w -= delta_w
    elif (start_w > delta_w and delta_w>0):
        start_w += delta_w
    else:
        start_w = 0
    print("x_min ", x_min, "x_max", x_max, "mid_x ",mid_x)
    print("delta x ", int(((x_min+x_max)/2-mid_x)))
    

    img1 = np.pad(source_img[:,:,0],pad_num,'symmetric')[start_h:start_h + h, start_w:start_w + w]
    img2 = np.pad(source_img[:,:,1],pad_num,'symmetric')[start_h:start_h + h, start_w:start_w + w]
    img3 = np.pad(source_img[:,:,2],pad_num,'symmetric')[start_h:start_h + h, start_w:start_w + w]

    source_img=np.concatenate([img1[:,:,np.newaxis],img2[:,:,np.newaxis],img3[:,:,np.newaxis]],2)




    print("source_img shape", source_img.shape, "crop_mask ", crop_mask.shape)
    crop_logo = source_img * crop_mask
    logo = target_img.copy()
    alpha = 0.8
    cv2.imwrite(f'./crop_logo_{source_name}.jpg', crop_logo)
    logo[x_min:x_min+h, y_min:y_min+w, :] = crop_logo
    cv2.imwrite(f'./logo_aligned_{source_name}.jpg', source_img)
    logo = logo.astype(np.float32) * target_img.astype(np.float32) / 255
    logo = logo.astype(np.float32) * alpha + target_img.astype(np.float32) * (1 - alpha)
    warp_img = ((logo * mask + (1 - mask) * target_img)).astype(np.uint8)
    # result = cv2.cvtColor(inv_standardization(cv2.cvtColor(warp_img, cv2.COLOR_BGR2HSV), base, scale, mu), cv2.COLOR_HSV2BGR)
    result = warp_img  # * mask + target_img * (1 - mask)
    cv2.imwrite(os.path.join(output_root, source_name.split('.')[0] + '_' + target_name.split('.')[0] + '.jpg'), result)


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
            for target in os.listdir(target_root):
                main(source_img_root=args.logo_root, target_img_root=target_root,
                     source_keypoint_path=args.logo_keypoint_root, target_keypoint_path=args.model_keypoint_root,
                     output_root=output_root, source_name=source, target_name=target, target_folder=target_folder)
