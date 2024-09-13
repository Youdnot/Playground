# README

文件目录结构

```
- Root Folder
	- 2404 Du_dataset
  - vr.net
  - README.md
  - requirements.txt
```

以上为试运行时的代码目录结构，如果需要

# Du_dataset

## 文件组织

- Dataset文件夹：数据集文件的放置位置
- eda：简单的数据总览
- video_metadata：提取生成视频的元数据，后续根据元数据对原始视频进行分割
- Subvideo：分割程序

# vr.net

## 文件组织

根据 Google Drive 链接提供的文件，数据集的部分组织方式如下：

单独的压缩包里是object和camera等坐标变化数据，以csv组织；在每个实验的文件夹压缩包中，有图像和zfp以及一个.csv文件。

### 图像

三种图像类型包括

- sRGB
- Depth
- Motion flow

首字母表示了图像的类型，如d s m

### zfp

 zfp 是一种用于表示多维浮点和整数数组的压缩格式。 zfp 提供压缩数组类，支持对各个数组元素的高吞吐量读写随机访问。

- [LLNL/zfp: Compressed numerical arrays that support high-speed random access](https://github.com/LLNL/zfp)
- [zfp 1.0.1 documentation — zfp 1.0.1 documentation](https://zfp.readthedocs.io/en/release1.0.1/)

## 进度

- [x] 图像数据重新组织为视频方便读取输入
- [ ] zfp 文件解压读取

## 问题

### zfp文件处理

库的安装可以使用`conda install zfp`从清华源镜像安装，无需重新编译，同时还需要另外安装python的绑定，使用`conda install zfpy`。

在文件解压过程中出现问题。使用`zfpy.decompress_numpy(_compressed_data_)`会产生报错`ValueError: Failed to read required zfp header`，表明数据没有标头，应该使用`zfpy._decompress(compressed_data, ztype, shape)`，但是该函数需要数据的维度作为输入，而文件内对该部分数据的信息没有明确说明。