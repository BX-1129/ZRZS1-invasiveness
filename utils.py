import os
import shutil
from pathlib import Path
from typing import Iterable
import logging
from datetime import datetime
LOG_DIR = os.path.join('log')
os.makedirs(LOG_DIR, exist_ok=True)

class NonHTTPFilter(logging.Filter):

    def filter(self, record):
        return 'HTTP Request' not in record.msg
logger = logging.root
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(asctime)s - %(filename)s:%(lineno)4s]\t%(levelname)s\t%(message)s'))
handler.addFilter(NonHTTPFilter())
logger.handlers = [handler]
DEBUG = True

def check_dir(save_dir, *input_dir):

    def check_ab(a, b):
        if b in a:
            return os.path.dirname(a) == os.path.dirname(b) and a != b
        return True
    for indir in [i for i in input_dir if i]:
        if len(save_dir) > len(indir):
            if not check_ab(save_dir, indir):
                return False
        elif not check_ab(indir, save_dir):
            return False
    return True

def get_pair_from_dir(path, x=None, y=None):
    items = os.listdir(path)
    x = x or 'images'
    y = y or 'masks'
    assert x in items and y in items
    images_path = Path(os.path.join(path, x))
    masks_path = Path(os.path.join(path, y))
    images = []
    masks = []
    for l_ in os.listdir(images_path):
        if not l_.startswith('.'):
            f_name, _ = os.path.splitext(l_)
            mask_file = list(masks_path.glob(f_name + '*'))
            if len(mask_file) == 1:
                images.append(os.path.abspath(os.path.join(images_path, l_)))
                masks.append(os.path.abspath(mask_file[0]))
    return (images, masks)

def get_pair_from_2dir(xpath, ypath, strict: bool=True):
    assert os.path.isdir(xpath) and os.path.isdir(ypath)
    images = []
    masks = []
    xpath = Path(xpath)
    ypath = Path(ypath)
    if strict:
        for l_ in os.listdir(xpath):
            if not l_.startswith('.') and os.path.isfile(os.path.join(xpath, l_)):
                f_name, _ = os.path.splitext(l_)
                mask_file = [str(p) for p in ypath.glob(f_name + '.*')]
                if len(mask_file) == 1:
                    images.append(os.path.abspath(os.path.join(xpath, l_)))
                    masks.append(os.path.abspath(mask_file[0]))
                elif os.path.join(ypath, l_) in mask_file:
                    images.append(os.path.abspath(os.path.join(xpath, l_)))
                    masks.append(os.path.abspath(os.path.join(ypath, l_)))
                else:
                    print(f'{l_}在{ypath}没有找到任何对应数据！')
    else:
        images = sorted([os.path.join(xpath, i) for i in os.listdir(xpath) if not i.startswith('.')])
        masks = sorted([os.path.join(ypath, i) for i in os.listdir(ypath) if not i.startswith('.')])
    assert len(images) == len(masks), '获取的图像和mask数量不匹配'
    return (images, masks)

def sin_sout_prompt(prompt=''):

    print(f'欢迎使用. {prompt}')
    input_dir = input('请输入您的待转化目录:').strip()
    while not os.path.isdir(input_dir):
        input_dir = input(f'您输入的目录（{input_dir}）不合法，请输入您的待转化目录:').strip()
    output_dir = input('请输入您的转化到目录:').strip()
    while not check_dir(output_dir, input_dir):
        output_dir = input('输出目录必须是与输入目录相互独立，并且没有互相包含，请重新输入:').strip()
    return (input_dir, output_dir)

def din_sout_prompt(prompt='', dir1='', dir2=''):

    print(f'欢迎使用. {prompt}')
    xpath = input(f'请输入您的{dir1}目录:').strip()
    while not os.path.isdir(xpath):
        xpath = input(f'您输入的目录（{xpath}）不合法，请输入您的{dir1}目录:').strip()
    ypath = input(f'请输入您的{dir2}目录:').strip()
    while '（可以为空）' not in dir2 and (not os.path.isdir(ypath)):
        ypath = input(f'您输入的目录（{ypath}）不合法，请输入您的{dir2}目录:').strip()
    output_dir = input('请输入您的转化到目录:').strip()
    while not check_dir(output_dir, xpath, ypath):
        output_dir = input('输出目录必须是与输入目录相互独立，并且没有互相包含，请重新输入:').strip()
    return (xpath, ypath, output_dir)

def din_sout_prompt_check(prompt='', dir1='', dir2='', check=None):
    if check is None:
        check = [False, True]
    if not isinstance(check, Iterable):
        check = [check] * 2

    print(f'欢迎使用. {prompt}')
    xpath = input(f'请输入您的{dir1}:').strip()
    while check[0] and (not os.path.isdir(xpath)):
        xpath = input(f'您输入的（{xpath}）不合法，请输入您的{dir1}:').strip()
    ypath = input(f'请输入您的{dir2}:').strip()
    while check[1] and '（可以为空）' not in dir2 and (not os.path.isdir(ypath)):
        ypath = input(f'您输入的（{ypath}）不合法，请输入您的{dir2}:').strip()
    output_dir = input('请输入您的转化到目录:').strip()
    while not check_dir(output_dir, xpath, ypath):
        output_dir = input('输出目录必须是与输入目录相互独立，并且没有互相包含，请重新输入:').strip()
    return (xpath, ypath, output_dir)

def min_sout_prompt(prompt='', dir1='', dir2='', atlast: int=1):
    """
    多输入路径参数，至少有1个输入路径
    Args:
        prompt: 功能提示
        dir1: 第一个目录的输入提示符
        dir2: 第二个目录的输入提示符
        atlast: dir1至少需要包括多少目录，默认为1.
    Returns:

    """

    print(f'欢迎使用. {prompt}')
    xpath = [input(f'请输入您的{dir1}：').strip(), input(f'请输入您的{dir1}(为空时退出)：').strip()]
    while xpath[-1] != '':
        xpath.append(input(f'请输入您的{dir1}(为空时退出)：').strip())
    remain_xpath = []
    for xp in xpath[:-1]:
        if os.path.exists(xp):
            remain_xpath.append(xp)
        else:
            print(f'您输入的：{xp}目录不存在，我们将忽略！')
    xpath = remain_xpath
    while len(xpath) < atlast:
        xpath.append(input(f'请输入您的{dir1}(为空时退出)：').strip())
        while xpath[-1] != '':
            print(xpath)
            xpath.append(input(f'请输入您的{dir1}(为空时退出)：').strip())
        remain_xpath = []
        for xp in xpath[:-1]:
            if os.path.exists(xp):
                remain_xpath.append(xp)
            else:
                print(f'您输入的：{xp}目录不存在，我们将忽略！')
        xpath = remain_xpath
    ypath = input(f'请输入您的{dir2}:').strip()
    while '（可以为空）' not in dir2 and (not os.path.isdir(ypath)):
        ypath = input(f'您输入的（{ypath}）不合法，请输入您的{dir2}:').strip()
    output_dir = input('请输入您的转化到目录：').strip()
    while not check_dir(output_dir, *xpath, ypath):
        output_dir = input('输出目录必须是与输入目录相互独立，并且没有互相包含，请重新输入：').strip()
    return (xpath, ypath, output_dir)

def smin_sout_prompt(prompt='', dir1='', atlast: int=1):
    """
    多输入路径参数，至少有1个输入路径, single multi input, single output
    Args:
        prompt: 功能提示
        dir1: 第一个目录的输入提示符
        dir2: 第二个目录的输入提示符
        atlast: dir1至少需要包括多少目录，默认为1.
    Returns:

    """

    print(f'欢迎使用. {prompt}')
    xpath = [input(f'请输入您的{dir1}：').strip(), input(f'请输入您的{dir1}(为空时退出)：').strip()]
    while xpath[-1] != '':
        xpath.append(input(f'请输入您的{dir1}(为空时退出)：').strip())
    remain_xpath = []
    for xp in xpath[:-1]:
        if os.path.exists(xp):
            remain_xpath.append(xp)
        else:
            print(f'您输入的：{xp}目录不存在，我们将忽略！')
    xpath = remain_xpath
    while len(xpath) < atlast:
        xpath.append(input(f'请输入您的{dir1}(为空时退出)：').strip())
        while xpath[-1] != '':
            print(xpath)
            xpath.append(input(f'请输入您的{dir1}(为空时退出)：').strip())
        remain_xpath = []
        for xp in xpath[:-1]:
            if os.path.exists(xp):
                remain_xpath.append(xp)
            else:
                print(f'您输入的：{xp}目录不存在，我们将忽略！')
        xpath = remain_xpath
    output_dir = input('请输入您的转化到目录：').strip()
    while not check_dir(output_dir, *xpath):
        output_dir = input('输出目录必须是与输入目录相互独立，并且没有互相包含，请重新输入：').strip()
    return (xpath, output_dir)

def fix_spacing(origin_img, w):
    w.SetOrigin(origin_img.GetOrigin())
    w.SetSpacing(origin_img.GetSpacing())
    w.SetDirection(origin_img.GetDirection())
    return w

def log_long_str(log_func, log_info):
    log_info_lines = str(log_info).split('\\n')
    for l in log_info_lines:
        log_func(l.strip('\\r'))

def delete_dir_if_exists(directory, directly=False, warning=False):
    """
    Delete directory if exists, If `directly`, delete without message prompt otherwise user should confirm.

    :param directory: Where to delete recursively.
    :param directly: Whether to delete directly.
    :param warning: WARNING INFO if ture
    :return:
    """
    if os.path.exists(directory):
        if not directly:
            logger.warning(f'{directory} already exists! Delete it? yes[y]/No[n]')
            i = input()
            if i.lower() == 'y' or i.lower() == 'yes':
                directly = True
        if directly:
            shutil.rmtree(directory, ignore_errors=True)
            logger.info(f'Successfully delete directory {directory}')
    elif warning:
        logger.warning(f'{directory} not exists!')

def create_directories_if_not_exists(*directories, truncate=False):
    """Create directories if not exists.

    :param directories: Directory to create.
    :param truncate: Truncate directory or not.
    """
    for directory in directories:
        if truncate:
            shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)

def create_dir_if_not_exists(directory, add_date=False, add_time=False) -> str:
    """Create directory if not exists.

    :param directory: Directory to create.
    :param add_date: Add date directory. If True, `directory/DATE` will be created.
    :param add_time: Add datetime directory. If True, `directory/DATETIME` will be created.
    :return path: The created path.
    """
    path = None
    if directory:
        path = directory
        if add_date:
            path = os.path.join(path, datetime.now().strftime('%Y%m%d'))
        if add_time:
            path = os.path.join(path, datetime.now().strftime('%H%M%S'))
        os.makedirs(path, exist_ok=True)
    return path

def create_img_msk_dir(root, img_='images', msk_='masks'):
    return (create_dir_if_not_exists(os.path.join(root, img_)), create_dir_if_not_exists(os.path.join(root, msk_)))

def truncate_dir(directory, del_directly=False, **kwargs):
    """
    Truncate directory.

    :param directory: Which directory to be truncated!
    :param del_directly:  Delete directory directly or not.
    :return:
    """
    delete_dir_if_exists(directory, del_directly)
    return create_dir_if_not_exists(directory, **kwargs)