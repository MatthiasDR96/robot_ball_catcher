# Imports
import numpy as np
import torch
import cv2
import ctypes
import pyrealsense2 as rs

# Plot bounding boxes
def plot_boxes(results, frame, model):

	# Get labels and coördinates
	labels, cord = results

	# Get image shape
	x_shape, y_shape = frame.shape[1], frame.shape[0]

	# Loop
	count = 0
	for i in range(len(labels)):

		# Get coordinates of bounding box
		row = cord[i]

		# Supress
		if row[4] < 0.5: continue

		# Get pixel coordinates
		x1 = int(row[0]*x_shape)
		y1 = int(row[1]*y_shape)
		x2 = int(row[2]*x_shape)
		y2 = int(row[3]*y_shape)

		# Color of bounding box
		bgr = (0, 255, 0) 

		# Get the name of label index
		classes = model.names 

		# If a person
		if int(labels[i]) == 0: count += 1

		# Get font for the label
		label_font = cv2.FONT_HERSHEY_SIMPLEX 

		# Plot the boxes
		cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) 

		# Put a label over box.
		cv2.putText(frame, classes[int(labels[i])], (x1, y1), label_font, 0.9, bgr, 2) 

	# Count people
	cv2.putText(frame, "Number of visible visitors: " + str(count), (int(np.shape(frame)[1]/4), 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

	return frame

# Score frame
def score_frame(frame, model):

	# Check device
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(device)
	model.to(device)

	# Convert frame to tensor
	frame = [torch.tensor(frame)]

	# Run through network
	results = model(frame)

	# Retrieve labels en coordinates
	labels = results.xyxyn[0][:, -1].numpy()
	cord = results.xyxyn[0][:, :-1].numpy()

	return labels, cord

# Loop
def main():

	# Loop
	while True:

		# Get RGB frame from camera
		frames = pipeline.wait_for_frames()
		frame = frames.get_color_frame()
		frame = np.asanyarray(frame.get_data())
			
		# Reshape image
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		# Score the frame
		results = score_frame(gray, model) 

		# Plot the boxes
		frame = plot_boxes(results, frame, model) 

		# Resize image
		resized = cv2.resize(frame, screensize, interpolation=cv2.INTER_AREA)

		# Show frame
		cv2.imshow('Visitor detection', resized)
		cv2.waitKey(1)

		# Stop statement
		if cv2.waitKey(1) == 27 & 0xff:  # press 'ESC' to quit
			pipeline.stop()
			cv2.destroyAllWindows()
			break

if __name__ == '__main__':

	# Camera params
	color_resolution = (1920, 1080)
	depth_resolution = (1280, 720)
	frames_per_second = 30

	# Connect to realsense
	pipeline = rs.pipeline()

	# Config camera
	config = rs.config()
	config.enable_device('821212060746')
	config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, frames_per_second)
	config.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1], rs.format.bgr8, frames_per_second)
	
	# Start streaming
	pipeline.start(config)

	# Model
	model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

	# Get screensize
	screensize = 4096, 2160

	# Loop
	try:
		main()
	except:
		pipeline.stop()
		cv2.destroyAllWindows()