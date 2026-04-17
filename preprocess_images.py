"""图片预处理脚本：将CWT图片批量预处理并缓存为numpy memmap文件。

可独立运行（提前生成缓存），也可被训练脚本导入调用。

独立运行示例：
    python preprocess_images.py --cwt_root /ai/xzx/PPG-more/ppg_dalia_cwt/picture/ppg_dalia --cache_path /ai/xzx/ppg/images_cache.dat
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def preprocess_all_images(file_list, cwt_root, cache_path):
    """预处理所有图片，保存为numpy memmap文件。只在首次运行时执行，后续直接加载。

    memmap将数据存储在磁盘上，按需加载到内存，不会一次性占用34GB内存。
    读取memmap是直接的二进制数据访问，跳过了JPEG解码，速度远快于原始方式。

    Args:
        file_list: 图片文件名列表
        cwt_root: 图片所在目录
        cache_path: memmap缓存文件路径
    Returns:
        numpy memmap, shape=(n, 224, 224, 3), dtype=float32
    """
    n = len(file_list)
    shape = (n, 224, 224, 3)

    # 如果缓存已存在，直接加载
    if os.path.exists(cache_path):
        print(f"加载已有图像缓存: {cache_path}")
        return np.memmap(cache_path, dtype='float32', mode='r', shape=shape)

    # 首次运行：多线程并行预处理并写入memmap
    # PIL的JPEG解码底层用C库实现会释放GIL，各idx写入memmap不同位置无竞争
    print(f"首次运行，预处理 {n} 张图片并缓存到磁盘...")
    images = np.memmap(cache_path, dtype='float32', mode='w+', shape=shape)

    def _process_one(args):
        idx, fname = args
        img = Image.open(os.path.join(cwt_root, fname)).convert('RGB').resize((224, 224))
        images[idx] = np.array(img, dtype='float32') / 255.0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(_process_one, (idx, fname))
                   for idx, fname in enumerate(file_list)]
        for _ in tqdm(as_completed(futures), total=n):
            pass

    images.flush()
    # 以只读模式重新打开，避免意外修改
    return np.memmap(cache_path, dtype='float32', mode='r', shape=shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理CWT图片并缓存为memmap文件")
    parser.add_argument('--cwt_root', type=str, help='CWT图片目录', default="/home/lenovo/ppg-bspc/orginal_picture/ppg_dalia" )
    parser.add_argument('--cache_path', type=str, help='memmap缓存输出路径', default="/home/lenovo/ppg-bspc/ppg_dalia_wo.dat")
    args = parser.parse_args()

    file_list = os.listdir(args.cwt_root)
    file_list.sort()
    print(f"共 {len(file_list)} 张图片")
    preprocess_all_images(file_list, args.cwt_root, args.cache_path)
    print("预处理完成")
