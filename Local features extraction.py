import argparse, concurrent.futures, functools, logging, multiprocessing, os, traceback, SimpleITK as sitk, matplotlib, nibabel as nib, numpy as np, radiomics, yaml, time
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from radiomics import featureextractor
radiomics.logger.setLevel(logging.ERROR)
from utils import get_pair_from_2dir, DEBUG, din_sout_prompt

# 在 gen_roi_rad_features.py 中添加
import radiomics
from radiomics import firstorder, glcm, gldm, glrlm, glszm, ngtdm, shape


# 显式注册所有特征类
def register_all_features(extractor):
    """手动注册所有 Pyradiomics 特征类"""
    feature_classes = {
        'firstorder': firstorder.RadiomicsFirstOrder,
        'glcm': glcm.RadiomicsGLCM,
        'gldm': gldm.RadiomicsGLDM,
        'glrlm': glrlm.RadiomicsGLRLM,
        'glszm': glszm.RadiomicsGLSZM,
        'ngtdm': ngtdm.RadiomicsNGTDM,
        'shape': shape.RadiomicsShape,
    }

    for cls_name, cls in feature_classes.items():
        extractor.enableFeatureClassByName(cls_name)
        # 确保类被初始化并注册
        try:
            cls(None, None)  # 传入 dummy 图像和掩码
        except Exception:
            pass  # 忽略初始化错误，只要类被加载即可


# 在创建 extractor 后调用此函数
extractor = featureextractor.RadiomicsFeatureExtractor()
register_all_features(extractor)

def get_largest_slice(img3d, mask3d, mask_id_spec=1):
    """
    Get the slice with largest tumor area
    Args:
        img3d: Numpy array. The whole CT volume (3D)
        mask3d: Numpy array. Same size as img3d, binary mask with tumor area set as 1, background as 0
        mask_id_spec: Int
    Returns:
        img: Numpy array. The 2D image slice with largest tumor area
        mask: Numpy array. The subset of mask in the same position of sub_img
    """
    # area = np.sum((mask3d == mask_id_spec), axis=(1, 2))
    # area_index = np.argsort(area)[-1]
    # img = img3d[(area_index, None[:None], None[:None])]
    # mask = mask3d[(area_index, None[:None], None[:None])]
    # return (
    #  img, mask, area_index)
    area = np.sum((mask3d == mask_id_spec), axis=(1, 2))
    if np.sum(area) == 0:  # 检查是否有匹配的像素
        print("\t警告：在mask中没有找到指定的mask_id值！")
        return None, None, None
    area_index = np.argsort(area)[-1]
    img = img3d[area_index]
    mask = mask3d[area_index]
    return img, mask, area_index


def extract_feature_unit(sub_img, p, q, padding=2, settings=None):
    """
    Args:
        sub_img: Numpy array. The tumor area defined by mask
        p,q: Int. The index of central pixel
        padding: Int. Number of pixels padded on each side after extracting tumor
        settings: Extra settings for extract features.
    Returns:
        features_temp: Dict. A dictionary contains all the radiomic features with keys used in "pyradiomics"
    """
    mask = np.copy(sub_img)
    mask[(None[:None], None[:None])] = 0
    mask[((p - padding)[:p + padding + 1], (q - padding)[:q + padding + 1])] = 1
    img_ex = sitk.GetImageFromArray([sub_img])
    mask_ex = sitk.GetImageFromArray([mask])
    extractor = featureextractor.RadiomicsFeatureExtractor(settings)
    featureVector = extractor.execute(img_ex, mask_ex)
    features = {}
    feature_names = set()
    for featureName in featureVector.keys():
        f_type, c_name, f_name = featureName.split("_")
        if f_type == "diagnostics":
            continue
        feature_names.add(featureName)
        features[featureName] = float(featureVector[featureName])

    return (
     features, feature_names)


def extract_radiomic_features(sub_img, sub_mask, mask_id_spec: int=1, workers=10, settings=None):
    """

    Args:
        sub_img: Numpy array. The tumor area defined by mask
        sub_mask: Numpy array. Same size as sub_img, binary values, 1:tumor area; 0:background
        mask_id_spec: Int. Mask ID in masks
        workers: Int. Number of workers used to process. Only works when "parallel" set to be True
        settings: Extra settings for extract features.

    Returns:
        features: Dict. A dictionary contains all the radiomic features with keys used in "pyradiomics"
    """
    mask_features = None
    if workers > 1:
        ps, qs = [], []
        partial_extract_feature_unit = functools.partial(extract_feature_unit, sub_img, settings=settings)
        for p in range(len(sub_img)):
            for q in range(len(sub_img[0])):
                if sub_mask[p][q] == mask_id_spec:
                    ps.append(p)
                    qs.append(q)

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(partial_extract_feature_unit, ps, qs)
            for (features, feature_name), p, q in zip(results, ps, qs):
                if mask_features is None:
                    mask_features = np.zeros(list(sub_img.shape) + [len(feature_name)])
                mask_features[(p, q)] = [features[fn] for fn in sorted(feature_name)]

    else:
        for p in range(len(sub_img)):
            for q in range(len(sub_img[0])):
                if sub_mask[p][q] == mask_id_spec:
                    features, feature_name = extract_feature_unit(sub_img, p, q, padding=2, settings=settings)
                    if mask_features is None:
                        mask_features = np.zeros(list(sub_img.shape) + [len(feature_name)])
                    mask_features[(p, q)] = [features[fn] for fn in sorted(feature_name)]

    return (
     mask_features, sorted(feature_name))


# def extract_feature_unit3D(sub_img, p, q, r, padding=2, settings=None):
#     """
#     Args:
#         sub_img: Numpy array. The tumor area defined by mask
#         p,q: Int. The index of central pixel
#         padding: Int. Number of pixels padded on each side after extracting tumor
#         settings: Extra settings for extract features.
#     Returns:
#         features_temp: Dict. A dictionary contains all the radiomic features with keys used in "pyradiomics"
#     """
#     mask = np.zeros_like(sub_img)
#     left = max(p - padding, 0)
#     top = max(q - padding, 0)
#     front = max(r - padding, 0)
#     mask[(left[:p + padding + 1], top[:q + padding + 1], front[:r + padding + 1])] = 1
#     img_ex = sitk.GetImageFromArray(sub_img)
#     mask_ex = sitk.GetImageFromArray(mask)
#     extractor = featureextractor.RadiomicsFeatureExtractor(settings)
#     featureVector = extractor.execute(img_ex, mask_ex)
#     features = {}
#     feature_names = set()
#     for featureName in featureVector.keys():
#         f_type, c_name, f_name = featureName.split("_")
#         if f_type == "diagnostics":
#             continue
#         feature_names.add(featureName)
#         features[featureName] = float(featureVector[featureName])
#
#     return (
#      features, feature_names)

def extract_feature_unit3D(sub_img, p, q, r, padding=2, settings=None):
    """
    Args:
        sub_img: Numpy array. The tumor area defined by mask
        p,q,r: Int. The index of central pixel
        padding: Int. Number of pixels padded on each side after extracting tumor
        settings: Extra settings for extract features.
    Returns:
        features_temp: Dict. A dictionary contains all the radiomic features with keys used in "pyradiomics"
    """
    try:
        mask = np.zeros_like(sub_img)

        # 计算边界，确保不会超出数组范围
        p_min = max(p - padding, 0)
        p_max = min(p + padding + 1, sub_img.shape[0])
        q_min = max(q - padding, 0)
        q_max = min(q + padding + 1, sub_img.shape[1])
        r_min = max(r - padding, 0)
        r_max = min(r + padding + 1, sub_img.shape[2])

        # 创建兴趣区域
        mask[p_min:p_max, q_min:q_max, r_min:r_max] = 1

        img_ex = sitk.GetImageFromArray(sub_img)
        mask_ex = sitk.GetImageFromArray(mask)

        extractor = featureextractor.RadiomicsFeatureExtractor(settings)
        featureVector = extractor.execute(img_ex, mask_ex)

        features = {}
        feature_names = set()

        for featureName in featureVector.keys():
            parts = featureName.split("_", 2)  # 分割最多2次，以防名称中有额外的下划线
            if len(parts) < 3:
                continue  # 跳过格式不符合的特征名

            f_type, c_name, f_name = parts
            if f_type == "diagnostics":
                continue

            feature_names.add(featureName)
            features[featureName] = float(featureVector[featureName])

        return features, feature_names

    except Exception as e:
        print(f"\t在点 ({p},{q},{r}) 提取特征时出错: {e}")
        # 返回空特征和特征名称集合
        return {}, set()
# def extract_radiomic_features3D(sub_img, sub_mask, mask_id_spec: int=1, workers=10, settings=None, skip_threshold: int=None, total_rounds: int=1):
#     """
#
#     Args:
#         sub_img: Numpy array. The tumor area defined by mask
#         sub_mask: Numpy array. Same size as sub_img, binary values, 1:tumor area; 0:background
#         mask_id_spec: Int. Mask ID in masks
#         workers: Int. Number of workers used to process. Only works when "parallel" set to be True
#         settings: Extra settings for extract features.
#         skip_threshold: Voxel larger than this will be skipped!
#         total_rounds: -
#
#     Returns:
#         features: Dict. A dictionary contains all the radiomic features with keys used in "pyradiomics"
#     """
#     mask_features = None
#     ps, qs, rs = np.where(sub_mask == mask_id_spec)
#     if skip_threshold is not None:
#         if len(ps) > skip_threshold:
#             print(f"\t这个样本将被Skip，由于数据量过大（{len(ps)} > {skip_threshold}）!")
#             return (None, None)
#     print(f"\t一共有{len(ps)}点需要计算局部特征。")
#     for _ in range(total_rounds):
#         if workers > 1:
#             partial_extract_feature_unit3D = functools.partial(extract_feature_unit3D, sub_img, settings=settings)
#             with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
#                 results = executor.map(partial_extract_feature_unit3D, ps, qs, rs)
#                 for (features, feature_name), p, q, r in zip(results, ps, qs, rs):
#                     if mask_features is None:
#                         mask_features = np.zeros((list(sub_img.shape) + [len(feature_name)]), dtype=(np.float32))
#                     mask_features[(p, q, r)] = [features[fn] for fn in sorted(feature_name)]
#
#         else:
#             for p, q, r in zip(ps, qs, rs):
#                 features, feature_name = extract_feature_unit3D(sub_img, p, q, r, padding=2, settings=settings)
#                 if mask_features is None:
#                     mask_features = np.zeros((list(sub_img.shape) + [len(feature_name)]), dtype=(np.float32))
#                 mask_features[(p, q, r)] = [features[fn] for fn in sorted(feature_name)]
#
#     return (
#      mask_features, sorted(feature_name))

def extract_radiomic_features3D(sub_img, sub_mask, mask_id_spec: int = 1, workers=10, settings=None,
                                skip_threshold: int = None, total_rounds: int = 1):
    """
    Args:
        sub_img: Numpy array. The tumor area defined by mask
        sub_mask: Numpy array. Same size as sub_img, binary values, 1:tumor area; 0:background
        mask_id_spec: Int. Mask ID in masks
        workers: Int. Number of workers used to process. Only works when "parallel" set to be True
        settings: Extra settings for extract features.
        skip_threshold: Voxel larger than this will be skipped!
        total_rounds: -

    Returns:
        features: Dict. A dictionary contains all the radiomic features with keys used in "pyradiomics"
    """
    mask_features = None
    ps, qs, rs = np.where(sub_mask == mask_id_spec)
    if skip_threshold is not None:
        if len(ps) > skip_threshold:
            print(f"\t这个样本将被Skip，由于数据量过大（{len(ps)} > {skip_threshold}）!")
            return (None, None)
    print(f"\t一共有{len(ps)}点需要计算局部特征。")

    # 初始化一个空列表，用于存储特征名称
    all_feature_names = []

    for _ in range(total_rounds):
        if workers > 1:
            partial_extract_feature_unit3D = functools.partial(extract_feature_unit3D, sub_img, settings=settings)
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                results = executor.map(partial_extract_feature_unit3D, ps, qs, rs)

                for idx, ((features, feature_name), p, q, r) in enumerate(zip(results, ps, qs, rs)):
                    try:
                        # 检查feature_name是否为可迭代对象
                        if isinstance(feature_name, int) or not hasattr(feature_name, '__iter__'):
                            print(f"\t警告: 在点 ({p},{q},{r}) 处理时feature_name不是可迭代对象: {type(feature_name)}")
                            continue

                        # 收集所有特征名称，用于后续处理
                        if not all_feature_names and feature_name:
                            all_feature_names = sorted(feature_name)

                        # 初始化mask_features数组（如果需要）
                        if mask_features is None and all_feature_names:
                            mask_features = np.zeros((list(sub_img.shape) + [len(all_feature_names)]), dtype=np.float32)

                        # 赋值特征
                        if mask_features is not None and all_feature_names:
                            try:
                                mask_features[(p, q, r)] = [features.get(fn, 0.0) for fn in all_feature_names]
                            except Exception as e:
                                print(f"\t在点 ({p},{q},{r}) 赋值特征时出错: {e}")
                                # 确保此点有值（即使是0）
                                mask_features[(p, q, r)] = [0.0] * len(all_feature_names)

                        # 每处理1000个点打印一次进度
                        if idx % 1000 == 0 and idx > 0:
                            print(f"\t已处理 {idx}/{len(ps)} 个点...")
                    except Exception as e:
                        print(f"\t处理点 ({p},{q},{r}) 时出错: {e}")
                        continue
        else:
            # 单线程处理
            for idx, (p, q, r) in enumerate(zip(ps, qs, rs)):
                try:
                    features, feature_name = extract_feature_unit3D(sub_img, p, q, r, padding=2, settings=settings)

                    # 检查feature_name是否为可迭代对象
                    if isinstance(feature_name, int) or not hasattr(feature_name, '__iter__'):
                        print(f"\t警告: 在点 ({p},{q},{r}) 处理时feature_name不是可迭代对象: {type(feature_name)}")
                        continue

                    # 收集所有特征名称，用于后续处理
                    if not all_feature_names and feature_name:
                        all_feature_names = sorted(feature_name)

                    # 初始化mask_features数组（如果需要）
                    if mask_features is None and all_feature_names:
                        mask_features = np.zeros((list(sub_img.shape) + [len(all_feature_names)]), dtype=np.float32)

                    # 赋值特征
                    if mask_features is not None and all_feature_names:
                        try:
                            mask_features[(p, q, r)] = [features.get(fn, 0.0) for fn in all_feature_names]
                        except Exception as e:
                            print(f"\t在点 ({p},{q},{r}) 赋值特征时出错: {e}")
                            # 确保此点有值（即使是0）
                            mask_features[(p, q, r)] = [0.0] * len(all_feature_names)

                    # 每处理1000个点打印一次进度
                    if idx % 1000 == 0 and idx > 0:
                        print(f"\t已处理 {idx}/{len(ps)} 个点...")
                except Exception as e:
                    print(f"\t处理点 ({p},{q},{r}) 时出错: {e}")
                    continue

    # 如果没有成功处理任何特征
    if mask_features is None or not all_feature_names:
        print("\t警告: 未能成功提取任何特征!")
        return None, None

    return mask_features, all_feature_names

def get_img_msk_arr(ip, mp, mask_id_spec):
    img_arr = np.array(nib.load(ip).data_obj)
    msk_arr = np.array(nib.load(mp).data_obj)
    if mask_id_spec is not None:
        mask_zeros = np.zeros_like(msk_arr)
        mask_zeros[msk_arr == mask_id_spec] = 1
        msk_out_arr = mask_zeros * img_arr
    else:
        mask_zeros = np.zeros_like(msk_arr)
        mask_zeros[msk_arr != 0] = 1
        msk_out_arr = mask_zeros * img_arr
    return (
     img_arr, mask_zeros, msk_out_arr)


def locate_tumor(img, padding=2):
    """
    Locate and extract tumor from CT image using mask
    Args:
        img: Numpy array. The whole image
        padding: Int. Number of pixels padded on each side after extracting tumor
    Returns:
        top, bottom, left, right
    """
    top_margin = min(np.where(img == 1)[0])
    bottom_margin = max(np.where(img == 1)[0])
    left_margin = min(np.where(img == 1)[1])
    right_margin = max(np.where(img == 1)[1])
    return (
     top_margin - padding, bottom_margin + padding + 1, left_margin - padding, right_margin + padding + 1)


# def main(image_dir, mask_dir, output_dir, whole=False, mask_id_spec=1, param_file=None, num_process=1, use_3d=True, overwrite=False, skip_threshold=None):
#     """
#
#     Args:
#         image_dir: 输入图像目录
#         mask_dir: Mask文件目录
#         output_dir: 输出特征目录
#         whole: 是否使用全部截面化，默认False
#         mask_id_spec: Mask中指定id对应的区域，默认为1。
#         param_file: 特征配置文件路径
#         num_process: 并行度，默认为1
#         use_3d: 是否使用3D，慎用！！！
#         overwrite: 是否使用缓存机制
#         skip_threshold: 超过多少个点之后，跳过！
#
#     Returns:
#
#     """
#     if use_3d:
#         print("WARNING: 正在使用3D的特征，提取的特征文件可能会非常大，请注意磁盘消耗！！！")
#     ipath_list, mpath_list = get_pair_from_2dir(image_dir, mask_dir)
#     viz_save_dir = os.path.join(output_dir, "viz")
#     print(f"一共找到{len(ipath_list)}个样本")
#     os.makedirs(output_dir, exist_ok=True)
#     if param_file is not None and os.path.exists(param_file):
#         with open(param_file) as pf:
#             settings = yaml.load((pf.read()), Loader=(yaml.FullLoader))
#     else:
#         settings = None
#     total_rounds = int(os.getenv("ONEKEY_ROUNDS", 10))
#     for ipath, mpath in zip(ipath_list, mpath_list):
#         try:
#             print(f'正在处理{ipath}{"." * total_rounds}')
#             save_name = os.path.join(output_dir, os.path.splitext(os.path.basename(ipath))[0])
#             if os.path.exists(f"{save_name}.npy"):
#                 if not overwrite:
#                     print(f"\t使用缓存机制，跳过{ipath}.")
#                     continue
#             image = sitk.GetArrayFromImage(sitk.ReadImage(ipath))
#             mask = sitk.GetArrayFromImage(sitk.ReadImage(mpath))
#             if image.shape != mask.shape:
#                 print(f"\t{ipath}数据尺寸不匹配，跳过.")
#                 continue
#             else:
#                 image_slice, mask_slice, slice_idx = get_largest_slice(image, mask, mask_id_spec=mask_id_spec)
#                 if use_3d:
#                     mf, fns = extract_radiomic_features3D(image, mask, workers=num_process, mask_id_spec=mask_id_spec,
#                       settings=settings,
#                       skip_threshold=skip_threshold,
#                       total_rounds=total_rounds)
#                 else:
#                     mf, fns = extract_radiomic_features(image_slice, mask_slice, workers=num_process, mask_id_spec=mask_id_spec,
#                       settings=settings)
#             if mf is None:
#                 continue
#             mf = mf.astype(np.float32)
#             np.save(save_name, mf)
#             if viz_save_dir is not None:
#                 save_dir = os.path.join(viz_save_dir, f"{os.path.basename(ipath)}-{slice_idx}")
#                 os.makedirs(save_dir, exist_ok=True)
#                 for idx, fn in enumerate(fns):
#                     fig = plt.figure()
#                     if use_3d:
#                         mf_slice = mf[(slice_idx, None[:None], None[:None], idx)]
#                     else:
#                         mf_slice = mf[(None[:None], None[:None], idx)]
#                     if not whole:
#                         top, bottom, left, right = locate_tumor(mask_slice, mask_id_spec)
#                         mf_slice = mf_slice[(max(top, 0)[:bottom], max(left, 0)[:right])]
#                     plt.imshow(mf_slice, cmap="jet")
#                     plt.axis("off")
#                     plt.xticks([])
#                     plt.yticks([])
#                     plt.savefig((os.path.join(save_dir, f"{fn}.png")), bbox_inches="tight")
#                     plt.close()
#
#         except Exception as e:
#             try:
#                 DEBUG or traceback.print_exc()
#                 print(f"处理{ipath}遇到错误，{e}")
#             finally:
#                 e = None
#                 del e

def main(image_dir, mask_dir, output_dir, whole=False, mask_id_spec=1, param_file=None, num_process=1, use_3d=True,
         overwrite=False, skip_threshold=None):
    """
    Args:
        image_dir: 输入图像目录
        mask_dir: Mask文件目录
        output_dir: 输出特征目录
        whole: 是否使用全部截面化，默认False
        mask_id_spec: Mask中指定id对应的区域，默认为1。
        param_file: 特征配置文件路径
        num_process: 并行度，默认为1
        use_3d: 是否使用3D，慎用！！！
        overwrite: 是否使用缓存机制
        skip_threshold: 超过多少个点之后，跳过！
    """
    if use_3d:
        print("WARNING: 正在使用3D的特征，提取的特征文件可能会非常大，请注意磁盘消耗！！！")

    ipath_list, mpath_list = get_pair_from_2dir(image_dir, mask_dir)
    viz_save_dir = os.path.join(output_dir, "viz")
    print(f"一共找到{len(ipath_list)}个样本")
    os.makedirs(output_dir, exist_ok=True)

    if param_file is not None and os.path.exists(param_file):
        with open(param_file) as pf:
            settings = yaml.load(pf.read(), Loader=yaml.FullLoader)
    else:
        settings = None

    total_rounds = int(os.getenv("ONEKEY_ROUNDS", 1))

    for ipath, mpath in zip(ipath_list, mpath_list):
        try:
            print(f'正在处理{ipath}{"." * total_rounds}')
            save_name = os.path.join(output_dir, os.path.splitext(os.path.basename(ipath))[0])

            # 检查缓存
            if os.path.exists(f"{save_name}.npy"):
                if not overwrite:
                    print(f"\t使用缓存机制，跳过{ipath}.")
                    continue

            # 读取图像和掩码
            try:
                image = sitk.GetArrayFromImage(sitk.ReadImage(ipath))
                mask = sitk.GetArrayFromImage(sitk.ReadImage(mpath))
            except Exception as read_error:
                print(f"\t读取{ipath}或{mpath}时出错: {read_error}")
                continue

            # 检查尺寸是否匹配
            if image.shape != mask.shape:
                print(f"\t{ipath}数据尺寸不匹配，跳过.")
                continue

            # 检查mask中是否包含指定的mask_id
            unique_values = np.unique(mask)
            if mask_id_spec not in unique_values:
                print(f"\t警告：在mask中没有找到指定的mask_id值 {mask_id_spec}！跳过.")
                continue

            # 获取最大肿瘤切片
            try:
                image_slice, mask_slice, slice_idx = get_largest_slice(image, mask, mask_id_spec=mask_id_spec)
                if image_slice is None:
                    print(f"\t{ipath}没有找到指定的mask区域，跳过.")
                    continue
            except Exception as slice_error:
                print(f"\t获取最大肿瘤切片时出错: {slice_error}")
                continue

            # 提取特征
            try:
                if use_3d:
                    mf, fns = extract_radiomic_features3D(
                        image, mask,
                        workers=num_process,
                        mask_id_spec=mask_id_spec,
                        settings=settings,
                        skip_threshold=skip_threshold,
                        total_rounds=total_rounds
                    )
                else:
                    mf, fns = extract_radiomic_features(
                        image_slice, mask_slice,
                        workers=num_process,
                        mask_id_spec=mask_id_spec,
                        settings=settings
                    )
            except Exception as extract_error:
                print(f"\t提取特征时出错: {extract_error}")
                continue

            # 如果没有成功提取特征，跳过
            if mf is None:
                print(f"\t{ipath}没有成功提取特征，跳过.")
                continue

            if fns is None or len(fns) == 0:
                print(f"\t{ipath}没有提取到特征名称，跳过可视化")
                fns = []

            # 保存特征
            try:
                mf = mf.astype(np.float32)
                np.save(save_name, mf)
                print(f"\t特征已保存到 {save_name}.npy")
            except Exception as save_error:
                print(f"\t保存特征时出错: {save_error}")
                continue

            # 可视化特征
            if viz_save_dir is not None and len(fns) > 0:
                try:
                    # 创建保存目录
                    try:
                        save_dir = os.path.join(viz_save_dir, f"{os.path.basename(ipath)}-{slice_idx}")
                        os.makedirs(save_dir, exist_ok=True)
                    except Exception as dir_error:
                        print(f"\t创建可视化目录时出错: {dir_error}")
                        continue

                    # 处理每个特征
                    for idx, fn in enumerate(fns):
                        try:
                            # 创建图形
                            fig = plt.figure(figsize=(8, 6))

                            # 获取特征切片
                            try:
                                if use_3d:
                                    # 确保slice_idx有效
                                    valid_slice_idx = 0 if slice_idx is None else slice_idx
                                    valid_slice_idx = min(valid_slice_idx, mf.shape[0] - 1)

                                    # 确保idx有效
                                    valid_idx = min(idx, mf.shape[3] - 1)

                                    # 使用正确的索引
                                    mf_slice = mf[valid_slice_idx, :, :, valid_idx]
                                else:
                                    # 确保idx有效
                                    valid_idx = min(idx, mf.shape[2] - 1)

                                    # 使用正确的索引
                                    mf_slice = mf[:, :, valid_idx]
                            except Exception as slice_error:
                                print(f"\t获取特征切片时出错: {slice_error}")
                                continue

                            # 裁剪肿瘤区域（如果需要）
                            if not whole and mask_slice is not None:
                                try:
                                    top, bottom, left, right = locate_tumor(mask_slice, padding=2)
                                    # 确保边界有效
                                    top = max(0, top)
                                    bottom = min(mf_slice.shape[0], bottom)
                                    left = max(0, left)
                                    right = min(mf_slice.shape[1], right)

                                    # 检查边界是否有效
                                    if top < bottom and left < right:
                                        mf_slice = mf_slice[top:bottom, left:right]
                                    else:
                                        print(f"\t裁剪边界无效: top={top}, bottom={bottom}, left={left}, right={right}")
                                except Exception as tumor_error:
                                    print(f"\t定位肿瘤区域时出错: {tumor_error}，使用完整切片")

                            # 可视化
                            try:
                                plt.imshow(mf_slice, cmap="jet")
                                plt.colorbar(label=fn)
                                plt.title(fn)
                                plt.axis("off")
                                plt.xticks([])
                                plt.yticks([])

                                # 保存图像
                                plt.savefig(os.path.join(save_dir, f"{fn}.png"), bbox_inches="tight", dpi=150)
                                plt.close(fig)
                            except Exception as viz_error:
                                print(f"\t可视化特征 {fn} 时出错: {viz_error}")
                                plt.close(fig)

                        except Exception as feature_error:
                            print(f"\t处理特征 {fn} 时出错: {feature_error}")
                            if 'fig' in locals():
                                plt.close(fig)

                    print(f"\t可视化完成，结果保存在: {save_dir}")

                except Exception as viz_all_error:
                    print(f"\t可视化过程出错: {viz_all_error}，但特征已保存")

        except Exception as main_error:
            if DEBUG:
                traceback.print_exc()
            print(f"处理{ipath}遇到错误，{main_error}")
            continue


# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     parser = argparse.ArgumentParser("计算所有的数据的Cluster。")
#     parser.add_argument("--whole", default=False, action="store_true", help="是否使用全局归一化，默认False")
#     parser.add_argument("--mask_id_spec", default=1, type=int, help="Mask中指定id对应的区域，默认为1。")
#     parser.add_argument("--param_file", default=None, type=str, help="特征配置文件路径")
#     parser.add_argument("-j", "--num_process", dest="j", default=1, type=int, help="并行度，默认为1")
#     parser.add_argument("--use_2d", default=False, action="store_true", help="是否使用3D，慎用！！！")
#     parser.add_argument("--overwrite", default=False, action="store_true", help="是否使用缓存机制")
#     parser.add_argument("--skip_threshold", default=None, type=int, help="超过多少个点之后，跳过！")
#     args, unparsed = parser.parse_known_args()
#     args.use_3d = not args.use_2d
#
#     xpath, ypath, o_dir = din_sout_prompt("生成Mask下的对应ROI的特征", "图像数据目录", "Mask数据目录")
#     main(xpath, ypath, o_dir, whole=(args.whole), mask_id_spec=(args.mask_id_spec), param_file=(args.param_file),
#       num_process=(args.j),
#       use_3d=(args.use_3d),
#       overwrite=(args.overwrite),
#       skip_threshold=(args.skip_threshold))
#     input("转化完成，按任意键退出！")
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # 1. 禁用所有警告和日志
    import warnings

    warnings.filterwarnings("ignore")  # 禁用所有警告信息

    import logging

    logging.getLogger().setLevel(logging.CRITICAL)  # 设置全局日志级别为CRITICAL
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # 特别禁用radiomics的日志
    radiomics.logger.setLevel(logging.CRITICAL)

    # 2. 设置一个环境变量，告诉子模块不要打印验证日志
    os.environ["RADIOMICS_NOLOG"] = "true"


    # 4. 命令行参数处理
    parser = argparse.ArgumentParser("计算所有的数据的Cluster。")
    parser.add_argument("--whole", default=False, action="store_true", help="是否使用全局归一化，默认False")
    parser.add_argument("--mask_id_spec", default=1, type=int, help="Mask中指定id对应的区域，默认为1。")
    parser.add_argument("--param_file", default=None, type=str, help="特征配置文件路径")
    # 重定向标准输出，捕获和过滤不需要的消息
    parser.add_argument("-j", "--num_process", dest="j", default=1, type=int, help="并行度，默认为1")
    parser.add_argument("--use_2d", default=False, action="store_true", help="是否使用3D，慎用！！！")
    parser.add_argument("--overwrite", default=False, action="store_true", help="是否使用缓存机制")
    parser.add_argument("--skip_threshold", default=None, type=int, help="超过多少个点之后，跳过！")
    args, unparsed = parser.parse_known_args()
    args.use_3d = not args.use_2d

    import sys

    original_stdout = sys.stdout


    class FilteredStdout:
        def __init__(self, original_stdout):
            self.original_stdout = original_stdout
            self.filtered_terms = ['validation.valid', '_distutils_hack']

        def write(self, message):
            # 如果消息不包含任何被过滤的术语，则写入
            if not any(term in message for term in self.filtered_terms):
                self.original_stdout.write(message)

        def flush(self):
            self.original_stdout.flush()


    sys.stdout = FilteredStdout(original_stdout)

    try:
        # 获取输入路径
        xpath, ypath, o_dir = din_sout_prompt("生成Mask下的对应ROI的特征", "图像数据目录", "Mask数据目录")

        # 运行主函数
        main(xpath, ypath, o_dir, whole=(args.whole), mask_id_spec=(args.mask_id_spec), param_file=(args.param_file),
             num_process=(args.j),
             use_3d=(args.use_3d),
             overwrite=(args.overwrite),
             skip_threshold=(args.skip_threshold))

    finally:
        # 恢复标准输出
        sys.stdout = original_stdout

    input("转化完成，按任意键退出！")
