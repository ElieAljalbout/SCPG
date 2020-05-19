import numpy as np 
import pylab
import nest
import random
import math
from neural_phase_generator.NPG import *
from common.NestUtil import *
from common.Constants import *
from ReSuMe import *

np.set_printoptions(threshold='nan')
nest.SetKernelStatus({'local_num_threads': 8})
msd 		= 123457
N_vp 		= nest.GetKernelStatus(['total_num_virtual_procs'])[0]

nest.SetKernelStatus({'grng_seed' : msd+N_vp})
nest.SetKernelStatus({'rng_seeds' : range(msd+N_vp+1, msd+2*N_vp+1)})


class CPG:

	def __init__(self
				,preSetup
				,tonicInput
				,spikedetector=None
				,npg=None
				,motorNeuronsNr=10
				,popSize=500
				,loadWeights=False
				,trainModel=False
				,desiredSignals=[]
				,path=[None,None]
				,starterNode=None
				,maxExpectedCycletime=1000):

		self.preSetup 		= preSetup 				# function that sets up needed nodes out of the CPG, this is usefull in case resetkernel is used all pre cpg nodes can be recreated
		self.tonicInput 	= tonicInput 			# input from the brain controlling frequency and phase shifts of CPG
		self.loadWeights 	= loadWeights 			# boolean indicating if should load weights or not/ if yes path should be set
		self.trainModel 	= trainModel 			# boolean indicating whether the training should occur or not
		self.desiredSignals = desiredSignals 		# supervision for resume training
		self.path 			= path 					# path related to loading the model
		self.npg 			= npg 					# neural phase generator
		self.starterNode	= starterNode 			# node responsible for starting the acitivity
		self.spikedetector 	= spikedetector 		# spike detector.
		self.motorNeuronsNr = motorNeuronsNr 		# number of motor neurons
		self.popSize		= popSize 				# population size per phase/module
		self.sd 			= None 					# spike detector for inner use
		self.frequencies  	= [1/2.0,1/3.0,1/4.0] 	# frequencies of patterns / other are available of course ut this is only for test
		self.cycles 		= {}					# dictionary containing the calculated cycle time of a cpg per frequency
		self.maxExpectedCycletime=maxExpectedCycletime

	# todo consider using t to inhibit inhibiting networks instead of qs
	def buildNetwork(self):

		self.sd 					= nest.Create("spike_detector",params={"withtime": True,"withgid": True}) 
		self.spikedetector 		 	= nest.Create("spike_detector",params={"withtime": True,"withgid": True}) if not self.spikedetector or not (self.loadWeights or self.trainModel) else self.spikedetector		
		self.npg 	  				= NPG() if not self.npg or not (self.loadWeights or self.trainModel) else self.npg
		self.pfNetworks				= [] # array containing the pattern forming(pf) network consisting of subnetworks for every module
		self.pfInhibitingNetwork    = [] # array containing networks inhibiting the pf networks
		
		self.initNPG()
		self.initNConnectPFN()
			
		self.limitInhibitingNets() # makes inhibiting nets spike only during the phase which they belong to
		self.createNConnectMotorNeurons()
		nest.SetStatus(self.starterNode ,{"spike_times":[5.0]})
		nest.Connect(self.starterNode,self.npg.modules[self.npg.start].hNeuron,syn_spec={"weight":1800.0})

		if self.trainModel: self.calculateCycles()
		if self.loadWeights or self.trainModel: self.updateNetworkModel()


	def initNPG(self):

		self.npg.create()
		self.npg.connectToTonicInput(self.tonicInput.drive)
		#self.npg.connectToSpikeDetector(self.spikedetector,[True,True,False,True])


	# Initialize and connect pattern forming network (PFN)
 	def initNConnectPFN(self):

 		for module in range(self.npg.modulesNr):

			self.pfNetworks.append(nest.Create(IAF_TYPE,self.popSize))
			self.pfInhibitingNetwork.append(nest.Create(IAF_TYPE,self.popSize))

			nest.Connect(self.pfNetworks[module],self.pfInhibitingNetwork[module],"one_to_one",{"weight":2000.0})
			nest.Connect(self.pfInhibitingNetwork[module],self.pfInhibitingNetwork[module],"one_to_one",{"weight":2000.0}) # auto synapse to ensure continuous spiking
			nest.Connect(self.pfInhibitingNetwork[module],self.pfNetworks[module],"one_to_one",{"weight":-4000.0})

			#nest.Connect(self.pfNetworks[module],self.spikedetector)
			#nest.Connect(self.pfInhibitingNetwork[module],self.spikedetector)
			self.connectPFN(module)


	def connectPFN(self,module):

		linksNr  	= int(self.npg.modules[module].interNr+1)
		miniPopSize	= self.popSize//linksNr
		miniPop 	= []
		for link in range(linksNr):

			miniPop=self.pfNetworks[module][link*miniPopSize:(link+1)*miniPopSize]
			nest.Connect([self.npg.modules[module].neurons[link]],miniPop,"all_to_all",{"weight": {"distribution": "uniform", "low": 50.0, "high": 1100.0}})
			nest.Connect(miniPop,miniPop,{"rule": "fixed_indegree", "indegree": 800},{"weight": {"distribution": "uniform", "low": -3.0, "high": -1.0}})


	# this function assures that the pfInhibitingNets are continuously running but only during their corresponding phases
	def limitInhibitingNets(self):

		for moduleI in range(self.npg.modulesNr):

			self.npg.modules[moduleI].inhibitModulesPopulation(self.pfInhibitingNetwork[moduleI])

	# create motor neurons, and connect PFN to them
	def createNConnectMotorNeurons(self):

		self.motorNeurons 		= nest.Create(IAF_TYPE,self.motorNeuronsNr)
		self.inhibitoryNeurons 	= []
		inhNeurons 				= []

		for idx,network in enumerate(self.pfNetworks):
			
			random.seed(Constants.rnSeeds[idx])
			mask=random.sample(range(self.popSize), self.popSize/5)
			inhNeurons=[network[i] for i in mask]
			excNeurons= np.delete(network,mask,0)
			self.inhibitoryNeurons=np.concatenate((self.inhibitoryNeurons,inhNeurons),axis=0)

			nest.Connect(inhNeurons,self.motorNeurons,"all_to_all",{"weight": {"distribution": "uniform", "low": -50.0, "high": -1.0}})
			nest.Connect(excNeurons.tolist(),self.motorNeurons,"all_to_all",{"weight": {"distribution": "uniform", "low": 1.0, "high": 5.0}})

			nest.Connect(self.motorNeurons,self.spikedetector)


	def calculateCycles(self):

		cycleTime=None
		for freq in self.frequencies:
			self.tonicInput.updateSpikeTimes(frequency=freq)
			cycleTime=self.getCPGCycle(self.maxExpectedCycletime)
			if  cycleTime!=-1:
				self.cycles[freq]=cycleTime

	# get the duration of one cycle of the CPG
	def getCPGCycle(self,expectedTime):

		firstSpikeNeuron = self.npg.modules[self.npg.start].hNeuron
		lastSpikeNeuron  = self.npg.modules[(self.npg.start+self.npg.modulesNr-1)%self.npg.modulesNr].tNeuron

		nest.Connect(firstSpikeNeuron,self.sd)
		nest.Connect(lastSpikeNeuron,self.sd)

		nest.Simulate(expectedTime+50)

		dSD = nest.GetStatus(self.sd,keys="events")[0]

		evs = dSD["senders"]
		ts 	= dSD["times"]

		ts, evs = zip(*sorted(zip(ts, evs),key=operator.itemgetter(0), reverse=False))

		timeFs=0 #time first spike of cpg
		for idx,val in enumerate(evs):
			if val==firstSpikeNeuron:
				timeFs=ts[idx]
				break

		timeLs1=0 # here we save the first time the tneuron of the last module spikes
		timeLs2=0 # here we save the first time the hneuron of the first module spike in the next cycle
		for idx,val in enumerate(evs):
			if  val==lastSpikeNeuron and timeLs1==0:
				timeLs1=ts[idx]
			if  val==firstSpikeNeuron and timeLs1!=0: # new cycle
				timeLs2=ts[idx]
				break
		
		timeLs=(timeLs2+timeLs1)/2 if timeLs2!=0 else timeLs1
		nest.ResetKernel()
		self.tonicInput,self.spikedetector,self.starterNode=self.preSetup()
		self.buildNetwork(False,False)
		if timeLs!=0: 
			return math.ceil(timeLs-timeFs)
		else:
			return -1

	def updateNetworkModel(self):

		if self.loadWeights: loadNetWeights(self.path[0],self.path[1])
		if self.trainModel:

			expectedTime 	= np.sum(self.npg.phasesDuration) + self.npg.modulesNr*10 # second term in for potential delay between multiple phases
			cycleTime 		= self.cycles[self.tonicInput.frequency]
			print(cycleTime,"cycleTime//////////////////",self.tonicInput.frequency)
			resume = ReSuMe(np.array(self.pfNetworks).ravel(),self.motorNeurons,self.tonicInput.updateFrequency,[self.tonicInput.frequency,cycleTime+50],list(self.inhibitoryNeurons.astype(int)),cycle=cycleTime)
			desiredSignalGlobal=[[self.motorNeurons[sig[0]],sig[1]] for sig in self.desiredSignals]
			resume.train(desiredSignalGlobal)