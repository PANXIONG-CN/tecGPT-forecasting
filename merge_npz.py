import os
import numpy as np
import glob
import tempfile
import shutil

# 基础路径
base_path = "processed_tec_data"
splits = ["train", "val", "test"]

for split in splits:
    print(f"处理 {split} 数据...")
    # 获取该分类下所有NPZ文件
    npz_files = sorted(glob.glob(os.path.join(base_path, split, f"{split}_*.npz")))

    if not npz_files:
        print(f"警告：{os.path.join(base_path, split)} 目录中未找到NPZ文件。")
        continue

    print(f"找到 {len(npz_files)} 个文件。")

    # 第一次扫描，计算样本总数和获取形状信息
    total_samples = 0
    x_shape = None
    y_shape = None

    for i, file_path in enumerate(npz_files):
        try:
            print(f"扫描文件 {i+1}/{len(npz_files)}: {os.path.basename(file_path)}")
            data = np.load(file_path)

            # 尝试获取数据键
            if "x" in data and "y" in data:
                x, y = data["x"], data["y"]
            elif "X" in data and "Y" in data:
                x, y = data["X"], data["Y"]
            else:
                keys = list(data.keys())
                if len(keys) >= 2:
                    x, y = data[keys[0]], data[keys[1]]
                else:
                    print(f"  警告：文件 {file_path} 中没有找到足够的数据键。")
                    continue

            # 记录形状信息和样本总数
            if x_shape is None:
                x_shape = x.shape[1:]  # 排除batch维度
                y_shape = y.shape[1:]

            total_samples += x.shape[0]
            print(f"  样本形状: {x.shape}, 累计样本数: {total_samples}")

        except Exception as e:
            print(f"  错误：处理文件 {file_path} 时出错: {str(e)}")

    if x_shape is None or y_shape is None:
        print(f"没有从 {split} 文件中获取到有效的形状信息，跳过。")
        continue

    # 创建临时内存映射文件
    print(f"创建内存映射文件以存储 {total_samples} 个样本...")

    # 为x和y数据创建内存映射文件
    x_filename = f"{split}_x.mmap"
    y_filename = f"{split}_y.mmap"

    # 定义最终数组形状
    x_final_shape = (total_samples,) + x_shape
    y_final_shape = (total_samples,) + y_shape

    # 计算数组元素数量和数据类型大小
    x_size = np.prod(x_final_shape)
    y_size = np.prod(y_final_shape)

    # 创建内存映射文件
    x_mmap = np.memmap(x_filename, dtype="float32", mode="w+", shape=x_final_shape)
    y_mmap = np.memmap(y_filename, dtype="float32", mode="w+", shape=y_final_shape)

    # 填充内存映射文件
    current_index = 0
    for i, file_path in enumerate(npz_files):
        try:
            print(f"处理文件 {i+1}/{len(npz_files)}: {os.path.basename(file_path)}")
            data = np.load(file_path)

            # 尝试获取数据键
            if "x" in data and "y" in data:
                x, y = data["x"], data["y"]
            elif "X" in data and "Y" in data:
                x, y = data["X"], data["Y"]
            else:
                keys = list(data.keys())
                if len(keys) >= 2:
                    x, y = data[keys[0]], data[keys[1]]
                else:
                    continue  # 已在上面的扫描中警告过

            # 将数据写入内存映射文件
            batch_size = x.shape[0]
            end_index = current_index + batch_size

            x_mmap[current_index:end_index] = x
            y_mmap[current_index:end_index] = y

            current_index = end_index
            print(f"  已写入 {batch_size} 个样本，总进度: {current_index}/{total_samples}")

            # 强制将缓冲区写入磁盘
            x_mmap.flush()
            y_mmap.flush()

        except Exception as e:
            print(f"  错误：写入文件 {file_path} 数据时出错: {str(e)}")

    # 打印合并后的形状
    print(f"合并后的x形状: {x_mmap.shape}, y形状: {y_mmap.shape}")
    print(f"总样本数: {total_samples} (检查是否与x的第一维度匹配: {x_mmap.shape[0]})")

    # 分批保存NPZ文件
    try:
        output_file = os.path.join(base_path, f"{split}.npz")
        print(f"保存合并文件到: {output_file}")

        # 创建临时目录保存分批数据
        temp_dir = tempfile.mkdtemp()
        batch_size = 5000  # 每批处理的样本数

        # 分批处理
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_x = x_mmap[batch_start:batch_end]
            batch_y = y_mmap[batch_start:batch_end]

            # 分批保存
            batch_filename = os.path.join(temp_dir, f"batch_{batch_start}_{batch_end}.npz")
            np.savez_compressed(batch_filename, x=batch_x, y=batch_y)
            print(f"  已保存批次 {batch_start}-{batch_end} / {total_samples}")

        # 合并所有批次文件
        print("合并所有批次文件...")
        with open(output_file, "wb") as outfile:
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                batch_filename = os.path.join(temp_dir, f"batch_{batch_start}_{batch_end}.npz")

                with open(batch_filename, "rb") as infile:
                    outfile.write(infile.read())

        print(f"已保存合并文件到: {output_file}\n")

        # 清理临时文件
        print("清理临时文件...")
        shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"保存 {split} 数据时出错: {str(e)}\n")

    # 关闭并删除内存映射文件
    del x_mmap
    del y_mmap

    # 删除临时的内存映射文件
    if os.path.exists(x_filename):
        os.remove(x_filename)
    if os.path.exists(y_filename):
        os.remove(y_filename)

print("所有数据处理完成！")
