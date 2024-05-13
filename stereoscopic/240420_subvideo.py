import cv2 as cv
import os
import time
import numpy as np
import pandas as pd

# 尝试先生成一个降帧率的原视频，再对原视频进行分割，看效率是否会有提升

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

def downsample_video_uneven(input_video_name, total_frames, input_fps, width, height, output_path, output_fps, ideal_input_length):
    """
    生成一个降帧率的原视频
    对于无法整除的特殊帧率重新编写逻辑，以1s为单位进行循环
    """
    cap = cv.VideoCapture(f"Dataset/videos/{input_video_name}.mp4")

    factor = int(input_fps // output_fps)    # 用于跳过帧的因子

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(f"{output_path}/{input_video_name}_downsample.mp4", fourcc, output_fps, (width, height))
    
    # 每一s中进行帧的跳过，余数帧直接丢弃
    count = 0
    i = 0

    while i < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if (i + 1) % factor == 0:
            out.write(frame)
            count += factor
            if count == output_fps*factor:
                i += (input_fps - count)
                count = 0
                # print(f"i = {i}, count = {count}")
        
        i += 1

        if i >= input_fps*ideal_input_length:
            print(f"input_fps: {input_fps}, ideal_input_length: {ideal_input_length}, i: {i}")
            print(f"frame range exceeds ideal total frames{input_fps*ideal_input_length}, break.")
            break


    out.release()
    cap.release()

def clip_subvideos(input_video_name, output_length, output_fps, output_width, output_height, random_shift_range:int):
    """
    从输入视频中剪切子视频
    """
    cap = cv.VideoCapture(f"{output_path}/{input_video_name}_downsample.mp4")

    downsample_video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    output_frames = output_length * output_fps

    # 循环创建子视频
    clip_count = 0
    start_frame = 0
    clip_count = downsample_video_frames - output_frames + 1

    print(f"Should generate {clip_count} clips.")

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
            shift_x = np.random.randint(-random_shift_range, (random_shift_range+1))
            shift_y = np.random.randint(-random_shift_range, (random_shift_range+1))
            
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

        break    # 测试用

    # 释放资源
    cap.release()
    # cv.destroyAllWindows()

    end_time = time.time()
    print(f"Processed {clip_count} clips from video {input_video_name}.")
    print(f"Cut Time: {end_time - start_time:.2f} seconds.")
    return clip_count


# hyperparameter
ideal_input_length = 60 # seconds

#  设置输出帧率和尺寸
output_length = 56 # seconds
output_fps = 10
output_width, output_height = 1280, 720
random_shift_range = 10

# 初始化收集裁剪数据的DataFrame
all_video_clips_dict = {}


if __name__ == "__main__":
    video_names = dir_initialization()
    for i in video_names:
        input_video_name = i
        print(f"Processing video {input_video_name}...")
        current_video_clips_dict = {}

        total_frames, input_fps, width, height, output_path = video_initialization(input_video_name)
        # downsample_video(input_video_name, total_frames, input_fps, width, height, output_path, output_fps)
        downsample_video_uneven(input_video_name, total_frames, input_fps, width, height, output_path, output_fps, ideal_input_length)
        clip_count = clip_subvideos(input_video_name, output_length, output_fps, output_width, output_height, random_shift_range)
        print(f"Video {input_video_name} Clip Done.")

        # 保存信息到字典中
        current_video_clips_dict["clip count"] = clip_count
        all_video_clips_dict[input_video_name] = current_video_clips_dict

    clip_df = pd.DataFrame(all_video_clips_dict)


        