import json
import os
import cv2
import numpy as np
import pickle
import argparse


def process_pickle(data):
    all_boxes = []
    #    a = []
    #    a.append(data[0])
    #    f = open('./data/bbox_result_test.pkl', 'wb')
    #    pickle.dump(a, f, 0)
    #    return None
    for n_img in range(len(data)):
        for i in range(13):
            for entry in data[n_img][0][i]:
                # print('img:%07d cat:%02d' % (n_img, i+1))
                # input()
                if entry is not None:
                    x1, y1, x2, y2 = entry[0:4]
                    w = x2 - x1
                    h = y2 - y1
                    box = dict()
                    box['image_id'] = n_img + 1
                    box['bbox'] = [int(x1), int(y1), int(w), int(h)]
                    box['score'] = float(entry[4])
                    box['category_id'] = i + 1
                all_boxes.append(box)
    with open('./data/bbox_result_test.json', 'w') as f:
        json.dump(all_boxes, f)
    #    print(len(all_boxes))
    return all_boxes


def read_pickle():
    with open('./data/bbox_result_test.pkl', 'rb') as f:
        raw_data = pickle.load(f)
        # all_boxes = self._process_pickle(raw_data)
        all_boxes = process_pickle(raw_data)
        print(all_boxes)
    return all_boxes


def draw_boxes(all_boxes, root='./dataset'):
    os.makedirs('/data/hrnet/bbox_output', exist_ok=True)
    i = 0
    for box in all_boxes:
        i += 1
        image_path = box['image_id']
        print(image_path)
        img = cv2.imread(image_path)
        x1, y1, w, h = np.array(box['bbox']).astype(int)
        x2 = x1 + w
        y2 = y1 + h
        print(x1, x2, y1, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
        print(f'/data/hrnet/bbox_output/{i}.jpg')
        cv2.imwrite(f'/data/hrnet/bbox_output/{i}.jpg', img)


def visualize(img_root='', kp_root=' ', output_root='./color_output_img', json_root='./color_output_json', prefix=''):
    # if not os.path.exists(json_root):
    #     os.mkdir(json_root)
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    gt_class_keypoints_dict = {1: (0, 25), 2: (25, 58), 3: (58, 89),
                               4: (89, 128), 5: (128, 143), 6: (143, 158), 7: (158, 168),
                               8: (168, 182), 9: (182, 190), 10: (190, 219),
                               11: (219, 256), 12: (256, 275), 13: (275, 294)}

    with open(kp_root, 'r') as jsonf:
        results = json.load(jsonf)

    nums = len(results)
    print(nums)
    for i in range(nums):
        result = results[i]
        #        if i == 14:
        #            print(int(result['category_id']))
        category_id = int(result['category_id'])
        image_id = result['image_id']
        keypoints = result['keypoints']
        score = result['score']
        #        print(image_id)
        image_path = image_id
        img = cv2.imread(image_path)
        interval = gt_class_keypoints_dict[category_id]
        for j in range(interval[0], interval[1]):
            x = int(keypoints[j * 3])
            y = int(keypoints[j * 3 + 1])
            cv2.circle(img, (x, y), 4, [255, 0, 170], thickness=-1)
        cv2.imwrite(os.path.join(output_root, f'{prefix}_{i}.jpg'), img)


def generate_pickle(root='./dataset', output_path='', is_logo=True, seg_root=''):
    all_boxes = []
    cnt = 0
    for name in os.listdir(root):
        if is_logo:
            img_path = os.path.join(root, name)
            name = '.'.join(name.split('.')[:-1]) + '.png'
            seg_path = os.path.join(seg_root, name)
            # print(seg_path)
            img = cv2.imread(img_path, 0)
            h, w = img.shape
            cnt += 1
            y1 = 0
            y2 = h-1
            x1 = 0
            x2 = w-1
            box = {}
            box['image_id'] = img_path
            box['bbox'] = [x1, y1, x2 - x1, y2 - y1]
            box['score'] = 1
            box['category_id'] = 1
            all_boxes.append(box)
        else:
            for name2 in os.listdir(os.path.join(root, name)):
                img_path = os.path.join(root, name, name2)
                seg_path = os.path.join(seg_root, name, '.'.join(name2.split('.')[:-1]) + '.png')
                print(seg_path)
                img = cv2.imread(seg_path, 0)
                h, w = img.shape
                new_img = img.copy()
                # for label in seg2landmark.keys():
                #     new_img[img == label] = seg2landmark[label]
                img = new_img == 4
                x_list, y_list = np.nonzero(img)
                cnt += 1
                if len(x_list) == 0:
                    y1 = float(0)
                    y2 = w-1.0
                    x1 = float(0)
                    x2 = h-1.0
                else:
                    min_x = max(0, np.min(x_list) - 15)
                    max_x = min(h - 1, np.max(x_list) + 15)
                    min_y = max(0, np.min(y_list) - 15)
                    max_y = min(w - 1, np.max(y_list) + 15)
                    y1 = float(min_x)
                    y2 = float(max_x)
                    x1 = float(min_y)
                    x2 = float(max_y)
                box = {}
                box['image_id'] = img_path
                box['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                box['score'] = 1
                box['category_id'] = 1
                all_boxes.append(box)
    # print(len(all_boxes))
    print(cnt)
    draw_boxes(all_boxes, root.replace('label', 'img'))
    # print('qaq', output_path)
    with open(output_path, 'w') as f:
        json.dump(all_boxes, f)


def generate_bbox_from_landmark(lanmark_path='', output_path=''):
    all_boxes = []
    with open(lanmark_path, 'r') as f:
        results = json.load(f)
    for i in range(len(results)):
        result = results[i]
        category_id = int(result['category_id'])
        image_id = result['image_id']
        keypoints = result['keypoints']
        keypoints = np.array(keypoints).reshape(-1, 3)[:25, :2]
        x1, y1 = np.maximum(keypoints.min(axis=0) - 10, np.array([0, 0]))
        x2, y2 = keypoints.max(axis=0) + 10
        box = dict()
        box['image_id'] = image_id
        box['bbox'] = [x1, y1, x2 - x1, y2 - y1]
        box['score'] = 1
        box['category_id'] = 1
        all_boxes.append(box)
    print(len(all_boxes))
    # draw_boxes(all_boxes)
    with open(output_path, 'w') as f:
        json.dump(all_boxes, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_big_bbox', action='store_true', help='generate a rough bbox.')
    parser.add_argument('--logo', action='store_true', help='generate a rough bbox.')
    parser.add_argument('--dataroot', type=str, default='root', help='name of the dataset.')
    parser.add_argument('--segroot', type=str, default='root', help='name of the dataset.')
    parser.add_argument('--outputroot', type=str, default='./data/bbox_result_test.json', help='name of the dataset.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # print('qaq', args.dataroot)
    if args.generate_big_bbox:
        generate_pickle(args.dataroot, output_path=args.outputroot, is_logo=args.logo, seg_root=args.segroot)
        # visualize(img_root=args.dataroot,
        #           kp_root='./output/deepfashion2/pose_hrnet/w48_384x288_adam_lr1e-3/model/results/keypoints_test_results_0.json',
        #           output_root='./visualize_landmark', prefix='logo' if args.logo else 'model')
    else:
        if args.logo:
            generate_bbox_from_landmark(
                lanmark_path='./output/deepfashion2/pose_hrnet/w48_384x288_adam_lr1e-3/logo/results/keypoints_test_results_0.json'
                , output_path='./data/bbox_result_test.json')
            visualize(img_root=args.dataroot,
                      kp_root='./output/deepfashion2/pose_hrnet/w48_384x288_adam_lr1e-3/logo/results/keypoints_test_results_0.json',
                      output_root='./visualize_landmark', prefix='logo')
        else:
            generate_bbox_from_landmark(
                lanmark_path='./output/deepfashion2/pose_hrnet/w48_384x288_adam_lr1e-3/model/results/keypoints_test_results_0.json'
                , output_path='./data/bbox_result_test.json')
            visualize(img_root=args.dataroot,
                      kp_root='./output/deepfashion2/pose_hrnet/w48_384x288_adam_lr1e-3/model/results/keypoints_test_results_0.json',
                      output_root='./visualize_landmark', prefix='model')

    # generate_pickle()
#    generate_pickle('/home/ella/ZMO/ella/Warping/Data_preprocessing/train_label')
