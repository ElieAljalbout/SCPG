###########################################################################
# 		This class represents a tonic input coming from the MLR			  #
#		The tonic drive can me more realistically implemented			  #
# 		but since this not the focus of this research we omit 			  #
#		this details													  #
###########################################################################

import nest
import numpy as np


class TonicDrive:

	DEFAULT_DELAY = 50.0

	def __init__(self,frequency,neuronsNr,start=1.0,period=10000.0,spikeTimes=[],name=None):

		self.frequency 	= frequency 	# spiking frequency or rate
		self.neuronsNr	= neuronsNr 	# number of neurons in the drive
		self.start		= start			# spiking start time
		self.period 	= period 		# period for which the drive should keep spiking
		self.spikeTimes = list(spikeTimes) 	# offering the ability tp set spikeTimes at construction
		self.name		= name 			# name for the drive, optional

	
	def create(self):

		self.drive  		= nest.Create("spike_generator",self.neuronsNr)
		self.spikeDetector 	= nest.Create("spike_detector",params={"withtime": True,"withgid": True}) 

		if  len(self.spikeTimes)==0:
			step=1/self.frequency
			self.spikeTimes=np.arange(self.start,self.period,step)

		nest.SetStatus(self.drive,{"spike_times":self.spikeTimes})

		print("Created the tonic drive")

	
	def updateSpikeTimes(self,start=None,period=None,frequency=1/4.0):

		start 	= start  if start 	else self.start
		period 	= period if period 	else self.period

		if  start< nest.GetKernelStatus("time"):
			lag = nest.GetKernelStatus("time") - start
			print("WARNING: spike generator started with lag" + str(lag))

		step = 1/frequency
		self.spikeTimes = np.arange(start,period,step)	
		nest.SetStatus(self.drive,{"spike_times":self.spikeTimes})


	# Even if the expectedPeriod is too big, the spikes_times will be updated and next step
	# One should make sure it's not less then it should be
	def updateFrequency(self,frequency,expectedPeriod):

		self.frequency 		= frequency
		step 				= 1/self.frequency
		nestTime 			= nest.GetKernelStatus("time")
		self.spikeTimes 	= np.arange(nestTime,nestTime+expectedPeriod + TonicDrive.DEFAULT_DELAY,step)

		nest.SetStatus(self.drive,{"spike_times":self.spikeTimes})

