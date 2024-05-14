import os
import time
import random
import numpy as np
import pandas as pd

import cv2 as cv
import pygwalker as pyg

## 预处理

def dir_initialization():
    """
    读取所有视频样本名称
    """
    video_names = os.listdir("Dataset/videos")
    video_names = [name.split(".")[0] for name in video_names]
    print(video_names)
    return video_names


def video_initialization(input_video_name):
    """
    初始化视频读取信息
    """
    cap = cv.VideoCapture(f"Dataset/videos/{input_video_name}.mp4")

    # 获取视频的总帧数、帧率和尺寸
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    input_fps = np.ceil(cap.get(cv.CAP_PROP_FPS))    # 原始帧率不是整数(29.99),会对后面跳过帧运算造成影响,所以向上取整
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / input_fps

    print(f"Total frames: {total_frames}, FPS: {input_fps}, Frame size: {width}x{height}, Duration: {duration:.2f} seconds.")

    cap.release()
    
    # 新建目标目录
    output_path = f"subvideos/{input_video_name}"
    os.makedirs(output_path) if not os.path.exists(output_path) else None
    print(f"Output path created: {output_path}")

    return total_frames, input_fps, width, height, output_path


## 生成降帧率视频


def downsample_video(input_video_name, input_total_frames, input_fps, width, height, output_path, output_fps):
    """
    生成一个降帧率的原视频
    """
    cap = cv.VideoCapture(f"Dataset/videos/{input_video_name}.mp4")

    factor = int(input_fps // output_fps)    # 用于跳过帧的因子

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(f"{output_path}/{input_video_name}_downsample.mp4", fourcc, output_fps, (width, height))
    
    for i in range(input_total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if (i + 1) % factor == 0:
            # print(f"{i}, {(i + 1) % factor}")
            out.write(frame)

    out.release()
    cap.release()


def downsample_video_uneven(input_video_name, input_total_frames, input_fps, width, height, output_path, output_fps, ideal_input_length):
    """
    生成一个降帧率的原视频
    对于无法整除的特殊帧率重新编写逻辑，以1s为单位进行循环
    """
    cap = cv.VideoCapture(f"Dataset/videos/{input_video_name}.mp4")
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(f"{output_path}/{input_video_name}_downsample.mp4", fourcc, output_fps, (width, height))
    
    # 每一s中进行帧的跳过，余数帧直接丢弃
    factor = int(input_fps // output_fps)    # 用于跳过帧的因子
    count = 0    # 保存每1s中已处理的帧数量
    i = 0
    actual_frame_count = 0
    s = 0    # 用于计算当前是原视频第几秒，作为判断条件解决原帧率未奇数时的一些裁剪问题

    while True:
        ret, frame = cap.read()
        # opencv的read函数在最后一帧的处理上可能会有问题，导致提前结束
        # if not ret:
        #     break

        if (i + 1 - s*input_fps) % factor == 0:
            out.write(frame)
            count += factor
            actual_frame_count += 1
            
            if count == output_fps*factor:
                jump_frame = input_fps - count
                i += jump_frame    # 如果count已经达到了output_fps，说明已经处理完了1s的帧，该秒内其余帧跳过
                count = 0
                s += 1
                # print(f"Skip next {jump_frame} frames.")
                # print(f"i = {i}, count = {count}")
        
        i += 1
        

        if i >= input_total_frames:
            print(f"Frame range exceeds total frames {input_total_frames}, break.")
            break
        if i >= input_fps*ideal_input_length:
            print(f"input_fps: {input_fps}, ideal_input_length: {ideal_input_length}, i: {i}")
            print(f"Frame range exceeds ideal total frames {input_fps*ideal_input_length}, break.")
            break


    out.release()
    cap.release()

    downsample_video_frames = actual_frame_count

    return downsample_video_frames

## Clips裁剪

def clip_subvideos(input_video_name, output_length, output_fps, output_width, output_height, random_shift_range:int, random_shift_stride:int):
    """
    从输入视频中剪切子视频
    """
    cap = cv.VideoCapture(f"{output_path}/{input_video_name}_downsample.mp4")

    downsample_video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    output_frames = output_length * output_fps

    # 循环创建子视频
    clip_count = 0
    start_frame = 0
    clip_count_all = downsample_video_frames - output_frames + 1
    print(f"Downsampled Video has {downsample_video_frames} frames, output frames {output_frames}.")
    print(f"Should generate {clip_count_all} clips.")

    start_time = time.time()

    while True:
        clip_count += 1
        
        # 定义输出视频的codec和创建VideoWriter对象
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(f"{output_path}/{input_video_name[:4]}_{clip_count}.mp4", fourcc, output_fps, (output_width, output_height))
        
        # 设置起始帧位置
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        # 计算图像中心
        center_x = width // 2
        center_y = height // 2

        # 随机初始裁切偏移
        shift_x = np.random.randint(-random_shift_range, (random_shift_range+1))
        shift_y = np.random.randint(-random_shift_range, (random_shift_range+1))
        shift_stride = list(range(-random_shift_stride, random_shift_stride+1, 1))
        
        for i in range(output_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 在随机初始偏移位置的基础上连续随机偏移
            while True:
                shift_x_new =  shift_x + random.choice(shift_stride)
                shift_y_new =  shift_y + random.choice(shift_stride)

                if abs(shift_x_new) <= random_shift_range and abs(shift_y_new) <= random_shift_range:
                    shift_x = shift_x_new
                    shift_y = shift_y_new
                    break
                else:
                    continue

            
            # Crop 裁切中心矩形区域
            shifted_frame = cv.getRectSubPix(frame, (output_width, output_height), (center_x+shift_x, center_y+shift_y))

            # Resize
            # shifted_frame = cv.resize(shifted_frame, (output_width, output_height))
            
            # 写入帧到输出视频
            out.write(shifted_frame)

            # 显示帧
            # cv.imshow("shifted_frame", shifted_frame)
            # if cv.waitKey(1) & 0xFF == ord("q"):
            #     break

        # 滑动窗口
        start_frame += 1

        # 释放输出视频流
        out.release()

        if downsample_video_frames < start_frame + output_frames:
            break

        # break    # 测试用

    # 释放资源
    cap.release()
    # cv.destroyAllWindows()

    end_time = time.time()
    process_time = end_time - start_time
    
    if clip_count == clip_count_all:
        print(f"Processed all {clip_count_all} clips from video {input_video_name}.")
    else:
        print(f"Processed {clip_count} clips from video {input_video_name}, max clips should be {clip_count_all}.")

    print(f"Cut Time: {process_time:.2f} seconds.")
    return clip_count, clip_count_all, process_time


## Hyperparameters


# hyperparameter
ideal_input_length = 60 # seconds

#  设置输出帧率和尺寸
output_length = 56 # seconds
output_fps = 10
output_width, output_height = 1280, 720

# 设置裁剪随机偏移参数
random_shift_range = 10
random_shift_stride = 2    # 为1时上下两帧画面的移动可以视为连续的（仅相差1像素）

# 初始化收集裁剪数据的DataFrame
all_video_clips_dict = {}


## 执行


video_names = dir_initialization()
start_time = time.time()

for i in video_names:
    input_video_name = i
    print(f"Processing video {input_video_name}...")

    # if i == "helicoptercrash":
    if True:
        current_video_clips_dict = {}

        input_total_frames, input_fps, width, height, output_path = video_initialization(input_video_name)
        # downsample_video(input_video_name, input_total_frames, input_fps, width, height, output_path, output_fps)
        downsample_video_frames = downsample_video_uneven(input_video_name, input_total_frames, input_fps, width, height, output_path, output_fps, ideal_input_length)
        clip_count, clip_count_all, process_time = clip_subvideos(input_video_name, output_length, output_fps, output_width, output_height, random_shift_range, random_shift_stride)
        print(f"Video {input_video_name} Clip Done.")

        # 保存信息到字典中
        current_video_clips_dict["downsample video frames"] = downsample_video_frames
        current_video_clips_dict["clip count"] = clip_count
        current_video_clips_dict["clip count all"] = clip_count_all
        current_video_clips_dict["process time"] = process_time
        all_video_clips_dict[input_video_name] = current_video_clips_dict

end_time = time.time()
total_time = end_time - start_time
print(f"Total Time: {total_time:.2f} seconds.")


clip_df = pd.DataFrame(all_video_clips_dict)


clip_df.T.to_csv("clip_data.csv", index=True)