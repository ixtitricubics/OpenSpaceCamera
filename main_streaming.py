# import libraries
from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears import NetGear
import asyncio
import time 
# import library
from vidgear.gears.asyncio import NetGear_Async
import cv2, asyncio
import time 
# define various tweak flags
options = {"flag": 0, "copy": False, "track": False}
import multiprocessing as mp

videos={
    # "192.168.1.41":"/mnt/7DD0FA902253E55C/datasets/openspace/tracking/12082021/cam192.168.1.41_2021_12_08_12_57_04.avi",
    "192.168.1.101":"/mnt/7DD0FA902253E55C/datasets/openspace/tracking/12082021/cam192.168.1.101_2021_12_08_12_57_04.avi",
    "192.168.1.102":"/mnt/7DD0FA902253E55C/datasets/openspace/tracking/12082021/cam192.168.1.102_2021_12_08_12_57_04.avi",
    "192.168.1.103":"/mnt/7DD0FA902253E55C/datasets/openspace/tracking/12082021/cam192.168.1.103_2021_12_08_12_57_04.avi",
    "192.168.1.104":"/mnt/7DD0FA902253E55C/datasets/openspace/tracking/12082021/cam192.168.1.104_2021_12_08_12_57_04.avi",
}
# servers = [NetGear(
#     address="192.168.0.6",
#     port=videos[i][1],
#     protocol="tcp",
#     pattern=1,
#     logging=True,
#     **options
# ) for i in range(len(videos))]  



# # !!! define your own video source here !!!
# # Open any video stream such as live webcam
# # video stream on first index(i.e. 0) device
# streams = [cv2.VideoCapture(vid_path) for vid_path,_ in videos]


# def main(streams, servers):
#     # loop over stream until its terminated
#     while True:
#         try:
#             frames = []
#             for stream in streams:
#                 (grabbed, frame) = stream.read()
#                 # check if frame empty
#                 if not grabbed:
#                     break
#                 frames.append(frame)
#             if(len(frames) == len(servers)):
#                 for idx, server in enumerate(servers):
#                     server.send(frames[idx])
#             time.sleep(0.1)
#         except KeyboardInterrupt:
#             break
#     for stream in streams:    
#         # safely close video stream
#         stream.release()
#     for server in servers:
#         # safely close server
#         server.close()

if __name__ == "__main__":
    # main(streams, servers)
    import subprocess
    import cv2

    # In my mac webcamera is 0, also you can set a video file name instead, for example "/home/user/demo.mp4"
    caps = {}
    commands = {}
    for ip in videos:        
        rtmp_url = f"rtmp://localhost/myapp/{ip}"
        caps[ip] = cv2.VideoCapture(videos[ip])

        # gather video info to ffmpeg
        fps = int(caps[ip].get(cv2.CAP_PROP_FPS))
        width = int(caps[ip].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[ip].get(cv2.CAP_PROP_FRAME_HEIGHT))

        # command and params for ffmpeg
        command = ['ffmpeg',
                '-y',
                '-hide_banner',
                '-stream_loop', '-1',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width, height),
                '-r', str(fps//2),
                '-i', '-',
                '-c:v', 'libx264',
                '-g', '48',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv',
                rtmp_url]
        

        # using subprocess and pipe to fetch frame data
        p = subprocess.Popen(command, stdin=subprocess.PIPE)
        commands[ip] = p
    while True:
        try:
            frames = {}
            for ip in caps:
                ret, frame = caps[ip].read()
                if not ret:
                    print("frame read failed")
                    break
                frames[ip] = frame
            if(len(frames) == len(videos)):
                # write to pipe
                for ip in frames:
                    p = mp.Process(target=commands[ip].stdin.write, args=(frames[ip].tobytes(),))
                    p.start()
            time.sleep(2/fps)
        except KeyboardInterrupt:
            break
    for ip in caps:
        caps[ip].release()
