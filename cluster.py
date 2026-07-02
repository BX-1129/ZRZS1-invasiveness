import argparse, os
from typing import Union, List
import SimpleITK as sitk, numpy as np, pandas as pd
from prettytable import PrettyTable
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from utils import min_sout_prompt, get_pair_from_2dir

def fix_spacing(origin_img, w):
    origin = origin_img.GetOrigin()
    spacing = origin_img.GetSpacing()
    direction = origin_img.GetDirection()
    w.SetOrigin(origin)
    w.SetSpacing(spacing)
    w.SetDirection(direction)
    return w


def cluster(ori_dirs, m_list, save_dir, mask_id: int=None, n_clusters=5, norm: bool=True):
    data = []
    for mpath in m_list:
        print(f"Processing mask: {os.path.basename(mpath)}")
        mask_img = sitk.ReadImage(mpath)
        mask_arr = sitk.GetArrayFromImage(mask_img)
        if not len(mask_arr.shape) == 3:
            raise AssertionError("ROI的数据必须是3维数据。")
        else:
            if mask_id is None:
                mask_axis = np.where(mask_arr != 0)
            else:
                mask_axis = np.where(mask_arr == mask_id)
            (xx, yy, zz) = mask_axis
            
            # Check if mask contains any non-zero voxels
            if len(xx) == 0:
                print(f"警告: 掩码 {os.path.basename(mpath)} 没有非零像素值，跳过此掩码。")
                continue
                
            features = []
            for ori_dir in ori_dirs:
                ipath = os.path.join(ori_dir, os.path.basename(mpath))
                if not os.path.exists(ipath):
                    ipath = f"{os.path.splitext(ipath)[0]}.npy"
                
                print(f"  检查图像路径: {ipath}")
                if not os.path.exists(ipath):
                    print(f"  警告: 无法找到对应的图像文件 {ipath}，跳过此特征。")
                    continue
                    
                if ipath.endswith(".npy"):
                    img_arr = np.load(ipath)
                else:
                    img = sitk.ReadImage(ipath)
                    img_arr = sitk.GetArrayFromImage(img)
                
                feature = []
                # FIX: Remove the double negation and simplify the condition
                if mask_arr.shape == img_arr.shape:
                    print(f"  形状匹配: mask {mask_arr.shape} == image {img_arr.shape}")
                    for (x, y, z) in zip(xx, yy, zz):
                        feature.append(np.array(img_arr[(x, y, z)]))
                elif len(img_arr.shape) > 3 and mask_arr.shape == img_arr.shape[:3]:
                    print(f"  形状匹配(多通道): mask {mask_arr.shape} == image {img_arr.shape[:3]}")
                    for (x, y, z) in zip(xx, yy, zz):
                        feature.append(np.array(img_arr[(x, y, z)]))
                else:
                    print(f"  警告: 掩码形状 {mask_arr.shape} 与图像形状 {img_arr.shape} 不匹配，跳过此特征。")
                    continue

                if len(feature) > 0:  # Check if we collected any features
                    feature = np.array(feature)
                    feature = np.reshape(feature, (len(xx), -1))
                    features.append(feature)
                    print(f"  成功提取特征: 形状 {feature.shape}")
                else:
                    print(f"  警告: 没有从图像中提取到特征。")

            # Check if we've collected any features for this mask
            if len(features) > 0:
                try:
                    concat_features = np.concatenate(features, axis=1)
                    data.append(concat_features)
                    print(f"  成功为掩码 {os.path.basename(mpath)} 组合特征: 形状 {concat_features.shape}")
                except ValueError as e:
                    print(f"  错误: 无法组合特征: {e}")
                    print(f"  特征形状: {[f.shape for f in features]}")
            else:
                print(f"  警告: 掩码 {os.path.basename(mpath)} 没有有效特征，跳过此掩码。")

    # Check if we have any data to cluster
    if len(data) == 0:
        raise ValueError("没有有效的数据可以进行聚类。请检查您的图像和掩码路径，以及它们的形状是否匹配。")

    data = np.concatenate(data, axis=0).astype(np.float32)
    print(f"最终数据形状: {data.shape}")
    
    if norm:
        for c in range(data.shape[1]):
            min_val = np.min(data[:, c])
            max_val = np.max(data[:, c])
            if max_val > min_val:  # Avoid division by zero
                data[:, c] = (data[:, c] - min_val) / (max_val - min_val)
            else:
                data[:, c] = 0  # If all values are the same, set to 0
                print(f"警告: 特征 #{c} 的所有值都相同，无法进行归一化。")

    print(f"一共获取了{data.shape[0]}个数据点，{data.shape[1]}个特征刻画，正在使用Kmeans进行{n_clusters}类别聚类分析...")
    if not isinstance(n_clusters, (list, tuple)):
        n_clusters = [n_clusters]
        
    ch_scores = []
    for ncluster in n_clusters:
        print(f"\t正在进行{ncluster}聚类...")
        if len(n_clusters) == 1:
            save2 = save_dir
        else:
            save2 = os.path.join(save_dir, f"cluster_{ncluster}")
        os.makedirs(save2, exist_ok=True)
        
        # Make sure we have at least as many data points as requested clusters
        if data.shape[0] < ncluster:
            print(f"警告: 数据点数量 ({data.shape[0]}) 小于聚类数量 ({ncluster})，跳过此聚类配置。")
            continue
            
        model = KMeans(n_clusters=ncluster, random_state=0)
        model.fit(data)
        table = PrettyTable(["Cluster ID", "Voxel Number"])
        vc = np.array(pd.Series(model.labels_).value_counts(ascending=False).reset_index())
        try:
            ch_score = calinski_harabasz_score(data, model.labels_)
        except Exception as e:
            print(f"警告: 无法计算 Calinski-Harabasz 分数: {e}")
            ch_score = 0

        for (k_, v_) in vc:
            table.add_row([k_, v_])

        table.add_row(["Calinski-Harabasz score", ch_score])
        ch_scores.append((ncluster, ch_score))
        print(f"聚类的类别分布：\n{table}")
        cluster_idx = 0
        print(f"正在保存Habitat Cluster到{save2}...")
        for mpath in m_list:
            mask_img = sitk.ReadImage(mpath)
            mask_arr = sitk.GetArrayFromImage(mask_img)
            if mask_id is None:
                mask_axis = np.where(mask_arr != 0)
            else:
                mask_axis = np.where(mask_arr == mask_id)
            (xx, yy, zz) = mask_axis
            
            # Skip if mask has no non-zero voxels
            if len(xx) == 0:
                print(f"\t跳过 {os.path.basename(mpath)} (没有非零像素值)")
                continue
                
            new_mask_arr = np.zeros_like(mask_arr)
            if cluster_idx + len(xx) <= len(model.labels_):
                for (x, y, z) in zip(xx, yy, zz):
                    new_mask_arr[(x, y, z)] = model.labels_[cluster_idx] + 1
                    cluster_idx += 1
            else:
                print(f"\t警告: 聚类标签数量 ({len(model.labels_)}) 小于掩码中的非零像素数量 ({len(xx)})，跳过此掩码。")
                continue

            new_mask_img = sitk.GetImageFromArray(new_mask_arr)
            fix_spacing(mask_img, new_mask_img)
            print(f"\t正在保存{os.path.basename(mpath)}...")
            sitk.WriteImage(new_mask_img, os.path.join(save2, os.path.basename(mpath)))

    return ch_scores


def main(image_dirs, mask_dir, output_dir, num_cluster: Union[(int, List[int])]=3, separate: bool=False, norm: bool=True):
    """
    Args:
        image_dirs: 图像的目录List，可以是任意长度
        mask_dir: Mask文件目录
        output_dir: 输出目录
        num_cluster: 聚类的个数，可以是一个聚类的集合或者一个具体的整数
        separate: 是否每个样本单独聚类
        norm: 是否每个特征都进行0-1标准化

    Returns:
        None
    """
    # Convert single directory to list if needed
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]
        
    # Find common mask files across all image directories
    print(f"图像目录: {image_dirs}")
    print(f"掩码目录: {mask_dir}")
    
    m_list = None
    for xp in image_dirs:
        print(f"查找目录 {xp} 中的图像和掩码对...")
        (il, ml) = get_pair_from_2dir(xp, mask_dir)
        print(f"  在 {xp} 中找到 {len(il)} 个图像文件和 {len(ml)} 个掩码文件")
        if m_list is None:
            m_list = set(ml)
        else:
            m_list = m_list & set(ml)

    m_list = sorted(list(m_list))
    print(f"一共找到{len(m_list)}共同的样本, separate: {separate}, norm: {norm}")
    
    if len(m_list) == 0:
        raise ValueError("没有找到任何共同的掩码文件，请检查您的图像和掩码目录。")
        
    ch_scores = []
    if separate:
        for m in m_list:
            try:
                print(f"处理单独样本: {os.path.basename(m)}")
                sample_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(m))[0])
                os.makedirs(sample_dir, exist_ok=True)
                ch_score = cluster(image_dirs, [m], sample_dir, n_clusters=num_cluster, norm=norm)
                ch_scores.extend(ch_score)
            except Exception as e:
                print(f"处理样本 {os.path.basename(m)} 时出错: {e}")
    else:
        try:
            ch_score = cluster(image_dirs, m_list, output_dir, n_clusters=num_cluster, norm=norm)
            ch_scores.extend(ch_score)
        except Exception as e:
            print(f"聚类过程中出错: {e}")
            raise
            
    if len(ch_scores) > 0:
        ch_scores = pd.DataFrame(ch_scores, columns=["n_clusters", "ch_score"]).groupby("n_clusters").agg("mean").reset_index()
        table = PrettyTable()
        table.field_names = ["n_clusters", "ch_score"]
        for row in np.array(ch_scores):
            table.add_row(row)

        print(f"--------------- 平均Calinski-Harabasz分数为 ---------------\n{table}")
        return ch_scores
    else:
        print("警告: 没有成功完成任何聚类。")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser("计算所有的数据的Cluster。")
    parser.add_argument("--cluster", default=5, type=int, nargs="*", help="Habitat的聚类类别数")
    parser.add_argument("--separate", default=False, action="store_true", help="是否每个样本单独做生境聚类")
    parser.add_argument("--ori_value", default=False, action="store_true", help="是否每个特征使用原始数值，默认False，即对特征进行0-1标准化。")
    args = parser.parse_args()
    
    try:
        (xpath, ypath, o_dir) = min_sout_prompt("聚类分析工具，计算所有的数据的Cluster。", "Image数据目录", "Mask数据目录")
        ch = main(xpath, ypath, o_dir, num_cluster=(args.cluster), separate=(args.separate), norm=(not args.ori_value))
        print(ch)
        input("转化完成，按任意键退出！")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        input("程序执行出错，按任意键退出！")
