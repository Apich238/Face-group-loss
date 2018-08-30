import sys
import argparse
import os
import numpy as np
import threading
import dlib
import cv2
import math


detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(r'dlib_data\shape_predictor_68_face_landmarks.dat')

cv_models = ['haarcascade_frontalface_default.xml',
             'haarcascade_frontalface_alt.xml',
             'haarcascade_frontalface_alt2.xml',
             'haarcascade_frontalface_alt_tree.xml',
             'haarcascade_profileface.xml']
cv_classifiers = [cv2.CascadeClassifier(os.path.join(r'opencv_data', m)) for m in cv_models]

rects = []


def read_image(folder):
    im = cv2.imread(folder, cv2.IMREAD_COLOR)
    return im


def detect_face(gray):
    '''
    :param im:
    :return: (x,y,w,h)
    '''
    faces = detector(gray, 2)
    t = len(faces) > 0
    minsz = 90

    if len(faces) > 0:
        for f in faces:
            if min(f.right() - f.left(), f.bottom() - f.top()) >= minsz and abs(250 - f.left() - f.right()) + abs(
                    250 - f.top() - f.bottom()) < 50 and min(f.left(), f.top()) > 0:
                return (f.left(), f.top(), f.right() - f.left(), f.bottom() - f.top()), t

    for i, classifier in enumerate(cv_classifiers):
        faces = classifier.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=1,
                                            minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) > 0:
            for det in faces:
                if min(det[2], det[3]) >= minsz and abs(250 - det[0] - (det[0] + det[2])) + abs(
                        250 - det[1] - (det[1] + det[3])) < 50:
                    return (det[0], det[1], det[2], det[3]), t
    return (), t


def predict_shape(gray, face):
    shape = landmark_predictor(gray, dlib.rectangle(face[0], face[1], face[0] + face[2], face[1] + face[3]))

    coords = np.empty((68, 2), dtype=int)

    for i in range(68):
        coords[i] = [shape.part(i).x, shape.part(i).y]

    return coords


def make_transform(im, shape, outsz, margin, debug_draw):
    a0 = shape[27]
    b0 = shape[8]
    if debug_draw:
        cv2.circle(im, (int(a0[0]), int(a0[1])), 3, (255, 255, 0), -1)
        cv2.circle(im, (int(b0[0]), int(b0[1])), 3, (0, 0, 255), -1)
    # лицо выравнивается так, что в точке (0.5,0.3) находится переносица, в точке (0.5,0.9) - подбородок
    # 0 относится к оригиналу изображения, 1 - к изображению после преобразования
    # выполняется перенос, поворот и масштабирование
    a1 = np.asarray([margin + (outsz - 2 * margin) * 0.5, margin + (outsz - 2 * margin) * 0.3], dtype=float)
    b1 = np.asarray([margin + (outsz - 2 * margin) * 0.5, margin + (outsz - 2 * margin) * 0.90], dtype=float)
    # перенос расчитывается исходя из центров отрезков
    m0 = (a0 + b0) / 2
    m1 = (a1 + b1) / 2

    d0 = b0 - a0
    d1 = b1 - a1

    e0 = d0 / np.linalg.norm(d0)
    e1 = d1 / np.linalg.norm(d1)
    # угол поворота
    d_angle = -(math.acos(e1[0]) - math.acos(e0[0]))
    dl = np.linalg.norm(d1) / np.linalg.norm(d0)
    # матрица поворота
    lR = dl * np.asarray([[math.cos(d_angle), math.sin(d_angle)],
                          [-math.sin(d_angle), math.cos(d_angle)]], dtype=float)
    # перенос
    ms = m1 - np.matmul(lR, m0)
    # полная матрица преобразования
    mx = np.zeros(shape=[2, 3], dtype=float)
    mx[:, :2] = lR
    mx[:, 2] = ms

    imout = cv2.warpAffine(im, mx, (outsz, outsz), flags=cv2.INTER_CUBIC)
    # cv2.imshow('after',imout)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return imout


def process(im, out_sz, margin, debug_draw, defaultBox):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    face, t = detect_face(gray)
    rejected = len(face) == 0
    if rejected:
        face = defaultBox
    if debug_draw:
        if rejected:
            cv2.rectangle(im, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 0, 255), 2)
        else:
            if t:
                cv2.rectangle(im, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 2)
            else:
                cv2.rectangle(im, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)
    shape = predict_shape(gray, face)
    if debug_draw:
        for (x, y) in shape:
            cv2.circle(im, (x, y), 3, (0, 255, 255), -1)
    imout = make_transform(im, shape, out_sz, margin, debug_draw)
    return imout


mutex2 = threading.Lock()


def process_subfolder_slave(files, targetdir, output_size, margin, debug_draw, defaultBox,thread_i):
    for fname in files:
        im = read_image(fname)
        im = process(im, output_size, margin, debug_draw, defaultBox)
        cv2.imwrite(os.path.join(targetdir, os.path.basename(fname)), im)
        print('.', end='', flush=True)


def process_subfolder(srcdir, targetdir, output_size, process_threads, margin, debug_draw, defaultBox):
    imgtypes = ['jpg', 'png', 'jpeg', 'gif', 'bmp']
    files = [os.path.join(srcdir, fname) for fname in os.listdir(srcdir) if fname.split('.')[-1] in imgtypes]
    parts = np.array_split(files, min(process_threads, len(files)))
    if len(parts) == 1:
        process_subfolder_slave(parts[0], targetdir, output_size, margin, debug_draw, defaultBox)
    else:
        threads = [threading.Thread(target=process_subfolder_slave,
                                    args=(parts[i], targetdir, output_size, margin, debug_draw, defaultBox,i))
                   for i in range(len( parts))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()


def filter(inDir, outDir, outSZ, addMargin, threads, debugDraw, defaultBox):
    global rejects
    rejects = 0

    dirs = [os.path.join(inDir, f) for f in os.listdir(inDir) if
            os.path.isdir(os.path.join(inDir, f))]

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    for i, subdir in enumerate(dirs):
        targetdir = os.path.join(outDir, os.path.basename(subdir))
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)

        srcdir = os.path.join(inDir, os.path.basename(subdir))
        print('\nprocessing {}({}/{})'.format(subdir, i + 1, len(dirs)), end='', flush=True)
        process_subfolder(srcdir, targetdir, outSZ, threads, addMargin, debugDraw, defaultBox)
    fcs = np.asarray(rects)
    print('\n')
    print('done')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(rejects)


def main(args):
    filter(args.input_dir, args.output_dir, args.output_size, args.add_margin, args.process_threads, args.debug_draw,
           [args.default_x, args.default_y, args.default_w, args.default_h])


# default x y w h
# casia
# [58, 68, 133, 133]
# lfw
# [75, 82, 100, 100]

def args_parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=r'D:\datasets\lfw\images',
                        help='')
    # параметры на выходе
    parser.add_argument('--output_size', type=int, default=160,
                        help='размер квадрата с лицом на выходном изображении')
    parser.add_argument('--add_margin', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=r'D:\datasets\lfw-my-filter',
                        help='')
    # общие параметры
    parser.add_argument('--process_threads', type=int, default=1,
                        help='количество потоков обработки')
    parser.add_argument('--debug_draw', type=bool, default=False,
                        help='вывод отладочной информации на изображения')
    parser.add_argument('--default_x', type=int)
    parser.add_argument('--default_y', type=int)
    parser.add_argument('--default_w', type=int)
    parser.add_argument('--default_h', type=int)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(args_parse(sys.argv[1:]))
