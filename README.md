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
# Visualization
# create a visualization object
vis = Visualization(cam_names, num_cameras, show_width=show_width, show_height=show_height)

# start the visualization thread
vis.start()

frames = [frame] # it should have the same number of images as the cameras
# show the images.
vis.update_frames(frames)

</pre>

if frame is None then camera hasnt read any image yet.
