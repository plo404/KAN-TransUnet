import os
import numpy as np
import nibabel as nib  # 需要安装: pip install nibabel
from PIL import Image  # 需要安装: pip install pillow
import glob

def nii_to_png(nii_folder, png_folder):
    # 如果输出目录不存在，则创建
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    # 获取所有 .nii.gz 文件
    files = glob.glob(os.path.join(nii_folder, "*.nii.gz"))
    
    print(f"Found {len(files)} nii.gz files in {nii_folder}")

    for file_path in files:
        file_name = os.path.basename(file_path)
        name_without_ext = file_name.replace('.nii.gz', '').replace('.nii', '')
        
        # 1. 读取 nii 文件
        img_obj = nib.load(file_path)
        img_data = img_obj.get_fdata()
        
        # 2. 处理维度
        # 此时 img_data 可能是 (H, W) 或 (H, W, 1) 或 (1, H, W)
        img_data = np.squeeze(img_data) 
        
        # 3. 转换像素值
        # 预测结果通常是 0, 1, 2... 
        # 我们需要把它变成 0, 255 以便肉眼观察
        # 如果是二分类（0背景，1血管），乘以255即可
        img_data = img_data.astype(np.uint8) * 255
        
        # 4. 保存为 PNG
        # 由于 SimpleITK 保存时可能会旋转/翻转坐标轴，如果发现图是倒的，可以使用 np.rot90 或 np.flip
        img = Image.fromarray(img_data)
        save_path = os.path.join(png_folder, name_without_ext + ".png")
        img.save(save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    # === 修改这里的路径 ===
    # 你的 nii.gz 结果所在的文件夹
    nii_result_dir = '/home/ly_s/WORK/Segment/TransUNet-main/data/OCTA_3M/test/result1' 
    # (注意：上面的路径可能要根据实际生成的子文件夹调整，请去你的result文件夹里看一眼确切位置)
    
    # 你希望保存 png 的文件夹
    png_save_dir = '/home/ly_s/WORK/Segment/TransUNet-main/data/OCTA_3M/test/result1_png'
    # ====================

    nii_to_png(nii_result_dir, png_save_dir)