# !/usr/bin/env python

import numpy as np 
import pandas as pd
import rosbag
import rospy
import cv2
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from TimestampSynchronizer import TimestampSynchronizer
from ImageEffects import ImageEffects

import sys
sys.path.append("../")
import config

class DatasetCreator:


	def __init__(self, bagName):
		self.bagName = bagName
		self.ts = TimestampSynchronizer(self.bagName)
		self.drone_topic = "optitrack/drone"
		self.camera_topic = "bebop/image_raw"
		self.body_topic = "optitrack/hand"


	def Sync(self, delay):

		print("unpacking...")
		#unpack the stamps
		camera_stamps = self.ts.UnpackBagStampsSingle(self.camera_topic)
		optitrack_stamps = self.ts.UnpackBagStampsSingle(self.body_topic)
		drone_stamps = self.ts.UnpackBagStampsSingle(self.drone_topic )
		if((len(drone_stamps) < len(camera_stamps) ) or (len(optitrack_stamps) < len(camera_stamps))):
			print("Error:recording data corrupted. not enough MoCap stamps.") 
			return

		print("unpacked stamps")
		
		#get the sync ids 
		otherTopics = [optitrack_stamps, drone_stamps]
		sync_camera_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, otherTopics, delay)
		sync_optitrack_ids = sync_other_ids[0]
		sync_drone_ids = sync_other_ids[1]	
		print("synced ids")

		return sync_camera_ids, sync_optitrack_ids, sync_drone_ids

	def CalculateRelativePose(self, optitrack_msg, drone_msg):

		part_orient = optitrack_msg.pose.orientation
		drone_orient = drone_msg.pose.orientation
		part_pose = optitrack_msg.pose.position
		drone_pose = drone_msg.pose.position

		x = part_pose.x - drone_pose.x
		y = part_pose.y - drone_pose.y
		z = part_pose.z - drone_pose.z
		part_quaternion = (part_orient.x, part_orient.y, part_orient.z, part_orient.w)
		part_euler = euler_from_quaternion(part_quaternion)
		drone_quaternion = (drone_orient.x, drone_orient.y, drone_orient.z, drone_orient.w)
		drone_euler = euler_from_quaternion(drone_quaternion)
		yaw = part_euler[2] - drone_euler[2]
		#print("part_pose={}".format(part_pose))
		#print("drone_pose={}".format(drone_pose))

		return x, y, z, yaw

	def CreateBebopDataset(self, delay, isHand, datasetName):
	
		if isHand == True:
			self.body_topic = "optitrack/hand"
		else:
			self.body_topic = "optitrack/head"

		
		sync_bebop_ids, sync_optitrack_ids, sync_drone_ids = self.Sync(delay)
		optitrack_msgs = self.ts.GetMessages(self.body_topic)
		drone_msgs = self.ts.GetMessages(self.drone_topic)
		
		bridge = CvBridge()
		
		x_dataset = []
		y_dataset = []

		#read in chunks because memory is low
		bebop_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
		chunk_size = 1000
		chunks = (bebop_msgs_count/chunk_size) + 1
		for chunk in range(chunks):
			bebop_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

			for i in range(len(bebop_msgs)):
				cv_image = bridge.imgmsg_to_cv2(bebop_msgs[i])
				cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
				cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
				x_dataset.append(cv_image)		
	
				optitrack_id = sync_optitrack_ids[chunk * chunk_size + i]		
				drone_id = sync_drone_ids[chunk * chunk_size + i]
				bebop_id = sync_bebop_ids[chunk * chunk_size + i]
				#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))
				
				x, y, z, yaw = self.CalculateRelativePose(optitrack_msgs[optitrack_id], drone_msgs[drone_id])
				
				y_dataset.append([int(isHand), x, y, z, yaw])

		print("dataset ready x:{} y:{}".format(len(x_dataset), len(y_dataset)))
		df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset})
		print("dataframe ready")
		df.to_pickle(datasetName)


	def CreateHimaxDataset(self, delay, isHand, datasetName):
	
		if isHand == True:
			self.body_topic = "optitrack/hand"
		else:
			self.body_topic = "optitrack/head"

		
		optitrack_msgs = self.ts.GetMessages(self.body_topic)
		drone_msgs = self.ts.GetMessages(self.drone_topic)
		bridge = CvBridge()

		sync_bebop_ids, sync_optitrack_ids, sync_drone_ids = self.Sync(delay)
		
		x_dataset = []
		y_dataset = []
	
		gammaLUT = ImageEffects.GetGammaLUT(0.6)
		vignetteMask = ImageEffects.GetVignetteMask(config.himax_width, config.himax_width)

		#read in chunks because memory is low
		bebop_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
		chunk_size = 1000
		chunks = (bebop_msgs_count/chunk_size) + 1
		for chunk in range(chunks):
			bebop_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

			for i in range(len(bebop_msgs)):
				cv_image = bridge.imgmsg_to_cv2(bebop_msgs[i])
				# image transform
				cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
				cv_image = cv2.LUT(cv_image, gammaLUT)
				cv_image = cv2.GaussianBlur(cv_image,(5,5),0)
				cv_image = cv2.resize(cv_image, (config.himax_width, config.himax_height), cv2.INTER_AREA)
				cv_image = cv_image *  vignetteMask[40:284, 0:324]
				cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_NEAREST)
				x_dataset.append(cv_image)		
	
				optitrack_id = sync_optitrack_ids[chunk * chunk_size + i]		
				drone_id = sync_drone_ids[chunk * chunk_size + i]
				bebop_id = sync_bebop_ids[chunk * chunk_size + i]
				#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))
				
				x, y, z, yaw = self.CalculateRelativePose(optitrack_msgs[optitrack_id], drone_msgs[drone_id])
				
				y_dataset.append([int(isHand), x, y, z, yaw])

		print("finished transformed bebop")
		self.camera_topic = "himax_camera"
		himax_msgs = self.ts.GetMessages(self.camera_topic)
		sync_himax_ids, sync_optitrack_ids, sync_drone_ids = self.Sync(delay)		

		for i in range(len(himax_msgs)):
			cv_image = bridge.imgmsg_to_cv2(himax_msgs[i])
			cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
			x_dataset.append(cv_image)		

			optitrack_id = sync_optitrack_ids[i]		
			drone_id = sync_drone_ids[i]
			himax_id = sync_himax_ids[i]
			#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))
			
			x, y, z, yaw = self.CalculateRelativePose(optitrack_msgs[optitrack_id], drone_msgs[drone_id])
				
			y_dataset.append([int(isHand), x, y, z, yaw])

		print("dataset ready x:{} y:{}".format(len(x_dataset), len(y_dataset)))
		df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset})
		print("dataframe ready")
		df.to_pickle(datasetName)

		
		

		
		
	



