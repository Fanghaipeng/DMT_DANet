import sys
sys.path.insert(0,'..')
import cv2
import time
import numpy as np
import os
import collections
import sys
import logging
import pdb
from sklearn.metrics import roc_auc_score
from common.glob import _global_dict, update_glob
import zipfile
import hashlib
import requests
from tqdm import tqdm

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)


_model_sha1 = {name: checksum for checksum, name in [
    ('25c4b50959ef024fcc050213a06b614899f94b3d', 'resnet50'),
    ('2a57e44de9c853fa015b172309a1ee7e2d0e4e2a', 'resnet101'),
    ('0d43d698c66aceaa2bc0309f55efdd7ff4b143af', 'resnet152'),
]}

encoding_repo_url = 'https://hangzh.s3.amazonaws.com/'
_url_format = '{repo_url}encoding/models/{file_name}.zip'


def read_annotations_3(data_path):
    lines = map(str.strip, open(data_path).readlines())
    data = []
    for line in lines:
        sample_path, mask_path, label = line.split()
        label = int(label)
        data.append((sample_path, mask_path ,label))
    return data

def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...' % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]


def get_resnet_file(name, root='~/.torch/models'):
    file_name = '{name}-{short_hash}'.format(name=name, short_hash=short_hash(name))
    root = os.path.expanduser(root)

    file_path = os.path.join(root, file_name + '.pth')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            print('Mismatch in the content of model file {} detected.' +
                  ' Downloading again.'.format(file_path))
    else:
        print('Model file {} is not found. Downloading.'.format(file_path))

    if not os.path.exists(root):
        os.makedirs(root)

    zip_file_path = os.path.join(root, file_name + '.zip')
    repo_url = os.environ.get('ENCODING_REPO', encoding_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')


def get_model_file(name, root='~/.torch/models'):
    root = os.path.expanduser(root)
    file_path = os.path.join(root, name + '.pth')
    if os.path.exists(file_path):
        return file_path
    else:
        raise ValueError('Model file is not found. Downloading or trainning.')


def read_annotations(data_path):
    lines = map(str.strip, open(data_path).readlines())
    data = []
    for line in lines:
        sample_path, label = line.split()
        label = int(label)
        data.append((sample_path,label))
    return data


def str2bool(in_str):
    if in_str in [1, "1", "t", "True", "true"]:
        return True
    elif in_str in [0, "0", "f", "False", "false"]:
        return False


def get_same_element_index(ob_list, word):
    return [i for (i, v) in enumerate(ob_list) if v == word]


def transfer_img_lab(mask):
    return np.max(mask)
    # if np.max(mask) == 0:  # real
    #     return 0
    # else:
    #     return 1


def calculate_img_score(pd, gt):
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    acc =  (true_pos + true_neg) / (true_pos + true_neg + false_neg+ false_pos  + 1e-6)
    sen = true_pos / (true_pos + false_neg + 1e-6)
    spe = true_neg / (true_neg + false_pos + 1e-6)
    # f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    f1 = 2 * sen * spe / (sen + spe)
    return acc, sen, spe, f1, true_pos, true_neg, false_pos, false_neg


def caculate_f1iou(pd, gt):
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        f1, iou = 1.0, 1.0
        return f1, iou
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(pd, gt)
    union = np.logical_or(pd, gt)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    return f1, iou


def get_img_lab(pd, gt, step, step_num):
    pd_lab_list = []
    if np.max(gt) == 0:
        gt_lab = 0
    else:
        gt_lab = 1
    for i in range(step_num):
        th = step * i
        if np.sum(pd)/255 <= th:
            pd_lab_list.append(0)
        else:
            pd_lab_list.append(1)
    return pd_lab_list, gt_lab


def remove_small(img):
    #print('img',img.shape)
    contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i, cnt in enumerate(contours):
        if cnt.shape[0] / (img.shape[0] * img.shape[1]) < 6e-5 or cnt.shape[0] < 36:
            for ix in range(cnt.shape[0]):
                img[cnt[ix][0][1], cnt[ix][0][0]] = 0
    return img


def lcm(x, y):
    s = x*y
    while y: x, y = y, x % y
    return s//x


def cut_bbox(img, bbox):
    return img[bbox[0]:bbox[2], bbox[1]:bbox[3]]

def caculate_IOU(pred, gt):
    insert = pred * gt
    union = 1.0 * ((pred + gt) > 0)
    return np.sum(insert) / np.sum(union)

def get_train_paths(args):
    train_data_path = os.path.join(args.data_path, args.train_collection, "annotations",args.train_collection + ".txt")
    val_data_path = os.path.join(args.data_path, args.val_collection, "annotations", args.val_collection + ".txt")
    model_dir = os.path.join(args.data_path, args.train_collection, "models", args.val_collection, args.config_name, "run_%d" % args.run_id)
    return [model_dir, train_data_path, val_data_path]


class Normalize_3D(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
            Tensor: Normalized image.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class UnNormalize_3D(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
            Tensor: Normalized image.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Progbar(object):
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)




if __name__ == "__main__":
    seg_path = "/data/chenxinru/user_data/DEFACTO/train_att/mask1_6.png"
    mask_path = "/data/chenxinru/user_data/DEFACTO/train_att/mask1_2.png"
    seg = cv2.imread(seg_path, 0)
    mask = cv2.imread(mask_path, 0)
    f1, iou, auc = caculate_f1iou(seg, mask)
    print("f1 %.4f\tiou %.4f\tauc %.4f" % (f1, iou, auc))

    # mask = cv2.imread('')
    # img = cv2.imread('')
    # import matplotlib.pyplot as plt
    # ori_shape = img.shape
    # ori_mask = mask
    #
    # img = pad_img(img, big_size=256, small_size=96)
    # mask = pad_img(mask)
    # # cv2.imwrite("padded_mask.png", mask)
    #
    # print(mask.shape)
    # shift = (max_anchors_size-min_anchors_size) // 2
    # inputs_small_index, _ = img2patches(mask, ps=min_anchors_size, pad=False, shift=shift)
    # padded_img = pad_img(mask, big_size=256, small_size=96)
    #
    # print(len(inputs_small_index))
    # inputs_small = [cut_bbox(mask, input_small_index)[:, :, 0] for input_small_index in inputs_small_index]
    # fake_seg = patches2img(inputs_small, ori_shape[0], ori_shape[1], ps=min_anchors_size)
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(ori_mask)
    # plt.subplot(1, 2, 2)
    # plt.imshow(fake_seg)
    # plt.savefig("test.png")
