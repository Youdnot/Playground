import cv2 as cv
import os
import time
import numpy as np

# import torch

# 尝试先生成一个降帧率的原视频，再对原视频进行分割，看效率是否会有提升

# hyperparameter
input_video_name = "dunerovers"
#  设置输出帧率和尺寸
output_length = 56 # seconds
output_fps = 10
output_width, output_height = 1280, 720

def initialization(input_video_name):
    """
    初始化
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

def downsample_video(input_video_name, total_frames, input_fps, width, height, output_path, output_fps):
    """
    生成一个降帧率的原视频
    """
    cap = cv.VideoCapture(f"Dataset/videos/{input_video_name}.mp4")

    factor = int(input_fps // output_fps)    # 用于跳过帧的因子

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(f"{output_path}/{input_video_name}_downsample.mp4", fourcc, output_fps, (width, height))
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if (i + 1) % factor == 0:
            # print(f"{i}, {(i + 1) % factor}")
            out.write(frame)

    out.release()
    cap.release()
def clip_subvideos(input_video_name, output_length, output_fps, output_width, output_height):
    """
    从输入视频中剪切子视频
    """
    cap = cv.VideoCapture(f"{output_path}/{input_video_name}_downsample.mp4")

    downsample_video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    output_frames = output_length * output_fps

    # 循环创建子视频
    clip_count = 0
    start_frame = 0

    print(f"Should generate {downsample_video_frames - output_frames + 1} clips.")

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
        
        for i in range(output_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 随机偏移
            shift_x = np.random.randint(-10, 11)
            shift_y = np.random.randint(-10, 11)
            
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

        # 更新剩余的总帧数
        # 不重复剪切
        # start_frame += output_frames
        # 滑动窗口
        start_frame += 1

        # 释放输出视频流
        out.release()

        if total_frames < start_frame + output_frames:
            break

        # break    # 测试用

    # 释放资源
    cap.release()
    # cv.destroyAllWindows()

    end_time = time.time()
    print(f"Processed {clip_count} clips from video {input_video_name}.")
    print(f"Cut Time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    total_frames, input_fps, width, height, output_path = initialization(input_video_name)
    # downsample_video(input_video_name, total_frames, input_fps, width, height, output_path, output_fps)
    clip_subvideos(input_video_name, output_length, output_fps, output_width, output_height)
    print("Done.")