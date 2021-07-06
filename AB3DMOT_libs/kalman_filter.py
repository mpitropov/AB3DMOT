# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker(object):
	"""
	This class represents the internel state of individual tracked objects observed as bbox.
	"""
	count = 0
	def __init__(self, bbox3D, info):
		"""
		Initialises a tracker using initial bounding box.
		"""
		# print(bbox3D) [x,y,z,theta,l,w,h]
		# print(info)

		# define constant velocity model
		self.kf = KalmanFilter(dim_x=10, dim_z=7)       
		self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
		                      [0,1,0,0,0,0,0,0,1,0],
		                      [0,0,1,0,0,0,0,0,0,1],
		                      [0,0,0,1,0,0,0,0,0,0],  
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0],
		                      [0,0,0,0,0,0,0,1,0,0],
		                      [0,0,0,0,0,0,0,0,1,0],
		                      [0,0,0,0,0,0,0,0,0,1]])     

		self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
		                      [0,1,0,0,0,0,0,0,0,0],
		                      [0,0,1,0,0,0,0,0,0,0],
		                      [0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0]])

		# # with angular velocity
		# self.kf = KalmanFilter(dim_x=11, dim_z=7)       
		# self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
		#                       [0,1,0,0,0,0,0,0,1,0,0],
		#                       [0,0,1,0,0,0,0,0,0,1,0],
		#                       [0,0,0,1,0,0,0,0,0,0,1],  
		#                       [0,0,0,0,1,0,0,0,0,0,0],
		#                       [0,0,0,0,0,1,0,0,0,0,0],
		#                       [0,0,0,0,0,0,1,0,0,0,0],
		#                       [0,0,0,0,0,0,0,1,0,0,0],
		#                       [0,0,0,0,0,0,0,0,1,0,0],
		#                       [0,0,0,0,0,0,0,0,0,1,0],
		#                       [0,0,0,0,0,0,0,0,0,0,1]])     

		# self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
		#                       [0,1,0,0,0,0,0,0,0,0,0],
		#                       [0,0,1,0,0,0,0,0,0,0,0],
		#                       [0,0,0,1,0,0,0,0,0,0,0],
		#                       [0,0,0,0,1,0,0,0,0,0,0],
		#                       [0,0,0,0,0,1,0,0,0,0,0],
		#                       [0,0,0,0,0,0,1,0,0,0,0]])

		
		# self.kf.R[0:,0:] *= 10.   # measurement uncertainty

		# Note detection score is info[6]
		# Must be [x,y,z,theta,l,w,h] while info is in [h,w,l,x,y,z,rot_y]
		variances = (info[7:14] + 0.0)
		# variances[:] = (1 - info[6]) * 2.0
		self.kf.R[0][0] = variances[3]
		self.kf.R[1][1] = variances[4]
		self.kf.R[2][2] = variances[5]
		self.kf.R[3][3] = variances[6]
		self.kf.R[4][4] = variances[0]
		self.kf.R[5][5] = variances[1]
		self.kf.R[6][6] = variances[2]

		self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
		self.kf.P *= 10.

		# self.kf.Q[-1,-1] *= 0.01    # process uncertainty
		self.kf.Q[7:, 7:] *= 0.01
		# self.kf.Q[:7, :7] *= 0.40 # added
		# self.kf.Q[0][0] = 1.2 # h
		# self.kf.Q[1][1] = 0.175 # w
		# self.kf.Q[2][2] = 0.10 # l
		# self.kf.Q[3][3] = 1.0 # angle
		# self.kf.Q[4][4] = 0.25 # x
		# self.kf.Q[5][5] = 0.16 # y
		# self.kf.Q[6][6] = 1.4 # z
		# print(self.kf.Q)
		# exit()

		self.kf.x[:7] = bbox3D.reshape((7, 1))

		# print('')
		# print('R', self.kf.R)
		# print('')
		# print('P', self.kf.P)
		# print('')
		# print('Q', self.kf.Q)
		# print('')
		# print('x', self.kf.x)
		# print('')
		# exit()

		self.time_since_update = 0
		self.id = KalmanBoxTracker.count
		KalmanBoxTracker.count += 1
		self.history = []
		self.hits = 1           # number of total hits including the first detection
		self.hit_streak = 1     # number of continuing hit considering the first detection
		self.first_continuing_hit = 1
		self.still_first = True
		self.age = 0
		self.info = info        # other info associated

	def update(self, bbox3D, info): 
		""" 
		Updates the state vector with observed bbox.
		"""
		self.time_since_update = 0
		self.history = []
		self.hits += 1
		self.hit_streak += 1          # number of continuing hit
		if self.still_first:
			self.first_continuing_hit += 1      # number of continuing hit in the fist time

		######################### orientation correction
		if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
		if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

		new_theta = bbox3D[3]
		if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
		if new_theta < -np.pi: new_theta += np.pi * 2
		bbox3D[3] = new_theta

		predicted_theta = self.kf.x[3]
		if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
			self.kf.x[3] += np.pi       
			if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
			if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

		# now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
		if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
			if new_theta > 0: self.kf.x[3] += np.pi * 2
			else: self.kf.x[3] -= np.pi * 2

		#########################     # flip

		# Update the R
		new_R = self.kf.R
		variances = (info[7:14] + 0.0)
		# variances[:] = (1 - info[6]) * 2.0
		new_R[0][0] = variances[3]
		new_R[1][1] = variances[4]
		new_R[2][2] = variances[5]
		new_R[3][3] = variances[6]
		new_R[4][4] = variances[0]
		new_R[5][5] = variances[1]
		new_R[6][6] = variances[2]
		self.kf.update(bbox3D, R=new_R)

		if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
		if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
		self.info = info

	def predict(self):       
		"""
		Advances the state vector and returns the predicted bounding box estimate.
		"""
		self.kf.predict()      
		if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
		if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

		self.age += 1
		if (self.time_since_update > 0):
			self.hit_streak = 0
			self.still_first = False
		self.time_since_update += 1
		self.history.append(self.kf.x)
		return self.history[-1]

	def get_state(self):
		"""
		Returns the current bounding box estimate.
		"""
		return self.kf.x[:7].reshape((7, ))