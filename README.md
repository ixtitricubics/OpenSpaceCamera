# camera
camera reading


usage:
<pre>
# create camera object
cam_obj = Camera(camera_config, ip)

# start its threads
cam_obj.start()

# read img
frame = cam()

#write img to video file
cam.insert_frame_to_save(frame)

# finish reading
cam.stop()
</pre>

if frame is None then camera hasnt read any image yet.
