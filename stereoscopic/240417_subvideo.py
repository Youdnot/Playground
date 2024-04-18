import cv2 as cv
import os
import time
import numpy as np

input_video_name = "cartooncoaster"
cap = cv.VideoCapture(f"Dataset/videos/{input_video_name}.mp4")

# hyperparameter 设置输出帧率和尺寸
output_length = 10
output_fps = 10
output_frames = output_length * output_fps
output_width, output_height = 1280, 720

# 获取视频的总帧数、帧率和尺寸
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

print(f"Total frames: {total_frames}, FPS: {fps}, Frame size: {width}x{height}")

# 新建目标目录
output_path = f"subvideos/{input_video_name}"
os.makedirs(output_path) if not os.path.exists(output_path) else None

# 循环创建子视频
clip_count = 0
start_frame = 0

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
    
    # 只处理子视频长度的帧数
    for i in range(output_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # 随机偏移
        shift_x = np.random.randint(-10, 11)
        shift_y = np.random.randint(-10, 11)
        
        # 裁切中心矩形区域
        shifted_frame = cv.getRectSubPix(frame, (output_width, output_height), (center_x+shift_x, center_y+shift_y))

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

    if total_frames <= start_frame + output_frames:
        break

    if clip_count >= 100:
        break

# 释放资源
cap.release()
# cv.destroyAllWindows()

end_time = time.time()
print(f"Processed {clip_count} clips from video {input_video_name}.")
print(f"Cut Time: {end_time - start_time:.2f} seconds.")