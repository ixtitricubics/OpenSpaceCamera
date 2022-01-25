# import required libraries
from vidgear.gears import NetGear
import cv2

# define various tweak flags
options = {"flag": 0, "copy": False, "track": False}

videos=[
    # ["/mnt/7DD0FA902253E55C/datasets/openspace/tracking/12082021/cam192.168.1.41_2021_12_08_12_57_04.avi","5454"],
    ["/mnt/7DD0FA902253E55C/datasets/openspace/tracking/12082021/cam192.168.1.101_2021_12_08_12_57_04.avi","5455"],
    ["/mnt/7DD0FA902253E55C/datasets/openspace/tracking/12082021/cam192.168.1.102_2021_12_08_12_57_04.avi","5456"],
    ["/mnt/7DD0FA902253E55C/datasets/openspace/tracking/12082021/cam192.168.1.103_2021_12_08_12_57_04.avi","5457"],
    ["/mnt/7DD0FA902253E55C/datasets/openspace/tracking/12082021/cam192.168.1.104_2021_12_08_12_57_04.avi","5458"],
]

# Define Netgear Client at given IP address and define parameters 
# !!! change following IP address '192.168.x.xxx' with yours !!!
clients = [NetGear(
    address="192.168.0.6",
    port=videos[i][1],
    protocol="tcp",
    pattern=1,
    receive_mode=True,
    logging=True,
    **options
) for i in range(len(videos))]

# loop over
while True:
    frames = []
    for client in clients:
        # receive frames from network
        frame = client.recv()
        
        # check for received frame if Nonetype
        if frame is None:
            break
        frames.append(frame)

    for idx, frame in enumerate(frames):
        # Show output window
        cv2.imshow("Output Frame" + str(idx), cv2.resize(frame,(640,480)))

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close client
client.close()