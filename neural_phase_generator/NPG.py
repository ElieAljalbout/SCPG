# this python class is an implementation of  the NPG circuit related to the paper:
# Towards a general neural controller for quadrupedal locomotion
# Author: Elie Aljalbout

import nest
import numpy as np
from PhaseModule import *

IAF_TYPE	= "iaf_psc_alpha"


msd 		= 123456
N_vp 		= nest.GetKernelStatus(['total_num_virtual_procs'])[0]
nest.SetKernelStatus({'grng_seed' : msd+N_vp})

class NPG:

	def __init__(self,modulesNr=4,phasesDuration=[200,300,200,300],start=0):

		self.modulesNr		= modulesNr				# Number of phases		
		self.phasesDuration = phasesDuration 		# Duration of each phase
		self.start 			= start					# Which module should start (Id)
		self.neurons		= []
		self.modules 		= []


	def create(self,names=None):

		module=None
		for mod in range(self.modulesNr):

			if names:
				module=PhaseModule(self.phasesDuration[mod],self.start==mod,names[mod])
			else:
				module=PhaseModule(self.phasesDuration[mod],self.start==mod)

			module.create()
			self.modules.append(module)

		for index in range(self.modulesNr-1):

			self.modules[index].connectToNextModule(self.modules[index+1])
			self.modules[index].inhibitModules([x for i,x in enumerate(self.modules) if i!=index])

		self.modules[self.modulesNr-1].connectToNextModule(self.modules[0])
		self.modules[self.modulesNr-1].inhibitModules(self.modules[:self.modulesNr-1])
			

	def connectToSpikeDetector(self,spikeDetector,connectTo=[True,False,False,False]):

		for mod in range(self.modulesNr):

			self.modules[mod].connectToSpikeDetector(spikeDetector,connectTo)


	# tonic source could be a population or a single neuron
	def connectToTonicInput(self,tonicSource):

		for mod in range(self.modulesNr):

			nest.Connect(tonicSource,self.modules[mod].neurons[:-1], "all_to_all",{"weight":100.0})