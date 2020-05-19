import numpy as np 
import nest
import pylab
import operator 
import json
import signal
        
class ReSuMe:

	TRAINING_THRESHOLD		= 1000					# number of training sessions m
	PERFORMANCE_THRESHOLD 	= 8						# threshold for training performance to stop
	dt 						= 2 					# discretisation time in ms	
	filter_tau				= 3			 			# shouldn't be 0 - represent the learning window
	tau_d					= filter_tau 			# shouldn't be 0
	tau_l					= filter_tau 			# shouldn't be 0
	lw_a					= 6		 				# learning window constant
	independantChange		= 3						# weight change that happens even when no presynaptic spike occurs

	# when passing multiple populations in learning Neurons, make sure you make the list sorted, no need for 
	# explicit sorting since neurons within a population are sorted, just sort the list of populations :O(1)
	def __init__(self,learningPreSynNeurons,learningNeurons,resetFunction,resetParams,inhPreSynNeurons,cycle=800,learningInputNeurons=None):


		#self.npg					= npg # rythm generator
		self.learningNeurons		= list(learningNeurons)#.tolist()
		self.learningPreSynNeurons 	= list(learningPreSynNeurons)#.tolist()
		self.learningInputNeurons	= learningInputNeurons
		self.resetFunction			= resetFunction
		self.resetParams			= resetParams
		self.inhPreSynNeurons		= inhPreSynNeurons
		self.spikeDetector 			= nest.Create("spike_detector",params={"withgid": True, "withtime": True})
		self.cycle 					= cycle
		self.shouldStopNow			= False
		self.plotNow				= False
		self.logFlag				= False

		signal.signal(signal.SIGINT, self.signal_handler)
		np.set_printoptions(threshold='nan')
		nest.Connect(self.learningNeurons,self.spikeDetector)
		nest.Connect(self.learningPreSynNeurons,self.spikeDetector)

	def signal_handler(self,signal, frame):

		command = raw_input('\nWhat is your command?')
		if str(command).lower()=="stop":
			self.shouldStopNow 	= True
		elif str(command).lower()=="plot":
			self.plotNow		= True
		else:
			exec(command)


	def plotResults(self,pms,asses):

		dSD = nest.GetStatus(self.spikeDetector,keys="events")[0]

		evs = dSD["senders"]
		dts = dSD["times"]
		dts, evs = zip(*sorted(zip(dts, evs),key=operator.itemgetter(0), reverse=False))
		idx=len(dts)
		while dts[idx-1]>=((self.session)*self.cycle) and idx>0:
			idx-=1
		
		evs = evs[idx:len(evs):1]
		dts = dts[idx:len(dts):1]
		dts = [x-((self.session)*self.cycle) for x in dts]
		mask= []

		for idx in range(len(evs)):
			if  evs[idx]< self.learningNeurons[0] or evs[idx]> self.learningNeurons[len(self.learningNeurons)-1]:
				mask.append(idx)

		evs 	= np.delete(evs,mask,0)
		dts 	= np.delete(dts,mask,0)

		desiredTimes 	= [x[1] for x in self.desiredSignals]
		desiredSpikes	= [x[0] for x in self.desiredSignals]
		
		pylab.figure(1)
		pylab.plot( dts,evs, ".b")
		pylab.plot(desiredTimes,desiredSpikes,".r")
		pylab.xlabel('Time (ms)')
		pylab.ylabel('Neuron GID (NEST)')

		pylab.figure(2)
		pts=np.arange(0,self.session+1,1)
		pylab.plot( pts,pms, "-b")
		pylab.xlabel('Learning Session')
		pylab.ylabel('Performance Index')
		pylab.show()

		pylab.figure(3)
		pts=np.arange(0,self.session+1,1)
		pylab.plot( pts,asses, "-b")
		pylab.xlabel('Learning Session')
		pylab.ylabel('Average Spike Shift Error')
		pylab.show()

	def updateSpikeTrain(self,idx,inputTrain,learnTrain):

		neuronId=self.spikeSenders[idx]
		if  neuronId>= self.learningNeurons[0] and neuronId<=self.learningNeurons[len(self.learningNeurons)-1]:
			learnTrain[neuronId]+=1
		else:
			inputTrain[neuronId]+=1

	# detects if the training is oscillating by looking at the performance index
	def detectOscillations(self,pm):
		
		lastIndex=len(pm)-1
		return (pm[lastIndex]>pm[lastIndex-1])!=(pm[lastIndex-1]>pm[lastIndex-2])


	# creates a map with keys the param indices and values 0
	def zerosMap(self,indices):
		
		theList={}
		for ind in indices:
			theList[ind]=0
		return theList

	def updateSpikeDetection(self,complete):

		#get spikes from repective neurons
		dSD = nest.GetStatus(self.spikeDetector,keys="events")[0]
		evs = dSD["senders"]
		ts  = dSD["times"]

		ts, evs = zip(*sorted(zip(ts, evs),key=operator.itemgetter(0), reverse=False))

		self.spikeSenders 		= evs 	# which neuron caused the spike
		self.spikeTimes 		= ts	# when did the spike happen

		#print(self.spikeSenders)
		if  not complete:
			mask=[]
			for idx in range(len(self.spikeSenders)):
				if  self.spikeSenders[idx]< self.learningNeurons[0] or self.spikeSenders[idx]> self.learningNeurons[len(self.learningNeurons)-1]:
					mask.append(idx)

			#print(mask)
			self.spikeSenders 	= np.delete(self.spikeSenders,mask,0)
			self.spikeTimes 	= np.delete(self.spikeTimes,mask,0)
			np.floor(self.spikeTimes)

	def getSpikeTrains(self,time,simulationDelay=0):

		# inintialize array of spike trains for network neurons
		inputTrain= self.zerosMap(self.learningPreSynNeurons)
		learnTrain=	self.zerosMap(self.learningNeurons)

		self.updateSpikeDetection(True)

		for idx, val in enumerate(self.spikeTimes):
			if shouldAddToSpikeTrain(val,time,simulationDelay):	
				self.updateSpikeTrain(idx,inputTrain,learnTrain)
		return inputTrain,learnTrain

	# this function aims to get a spike train representation of the 
	# desired signals fed to the training algorithm (ReSuMe)
	def codeSignalsIntoTrain(self,desiredSignals,time):

		train=self.zerosMap(self.learningNeurons)

		for signal in desiredSignals:
			if shouldAddToSpikeTrain(signal[1],time):
				train[signal[0]]+=1

		return train

	def lowPassFilter(self,time):

		filterArray=self.zerosMap(self.learningNeurons)
		self.updateSpikeDetection(False)

		for idx, val in enumerate(self.spikeTimes):
			if val<=time and val>=self.session*self.cycle:	
				filterArray[self.spikeSenders[idx]]+=np.exp((val-time)/ReSuMe.filter_tau)

		filterArray=[filterArray[x] for x in filterArray]
		return np.asarray(filterArray)

	def desiredlowPassFilter(self,desiredSignals,time):

		filterArray=self.zerosMap(self.learningNeurons)
		for signal in desiredSignals:
			if signal[1]<=time:
				filterArray[signal[0]]+=np.exp((signal[1]-time)/ReSuMe.filter_tau)

		filterArray=[filterArray[x] for x in filterArray]
		return np.asarray(filterArray)


	def computePerformanceIndex(self,desiredSignals):

		time=self.session*self.cycle
		pm=0 # performance index
		while time<(self.session)*self.cycle +self.cycle :
			pm+=np.linalg.norm(self.lowPassFilter(time)-self.desiredlowPassFilter(desiredSignals,time))
			time+=self.dt
		return pm

	def computeAverageSpikeShiftError(self,desiredSignals):

		self.updateSpikeDetection(False)
		count 	= 0
		average = 0

		usedSpike	= [0 for x in self.spikeTimes if x>=self.session*self.cycle]
		spikeTimes	= [x for x in self.spikeTimes if x>=self.session*self.cycle]
		spikeSenders= [y for idx,y in enumerate(self.spikeSenders) if self.spikeTimes[idx]>=self.session*self.cycle]

		for signal in desiredSignals:
			minTimeDist=self.cycle*100
			minIdx=-1
			for idx,val in enumerate(spikeTimes):
				tDiff=abs(signal[1]-val)
				if usedSpike[idx]==0 and spikeSenders[idx]==signal[0] and tDiff<minTimeDist:
					minTimeDist=tDiff
					minIdx=idx
				if tDiff>=minTimeDist:break
			if minIdx!=-1 : 
				usedSpike[minIdx]=1
				count+=1
			average+=minTimeDist
		
		nonUsedNr=	(len(usedSpike)-np.count_nonzero(usedSpike))
		average+= nonUsedNr*self.cycle*100 # here we are adding error for each spike that happened and wasn't desired to.
		count  += nonUsedNr
		average=average/count if count!=0 else float("inf")
		return average

	# save the weights in json format in a file
	def saveWeights(self):

		print("Saving weights=====")
		incommingSynapses,synapsesWeights=self.getIncommingWeights(self.learningNeurons,[],False,True)

		fw = open("weights/Weights-"+str(ReSuMe.filter_tau)+"-"+str(ReSuMe.independantChange)+"-"+str(ReSuMe.lw_a),'w')
		fs = open("weights/synapses-"+str(ReSuMe.filter_tau)+"-"+str(ReSuMe.independantChange)+"-"+str(ReSuMe.lw_a),'w')
		
		incommingSynapses = [[val[0],val[1],val[2],val[3],val[4]] for val in  incommingSynapses]
		synapsesWeights   = [w for w in synapsesWeights]

		json.dump(incommingSynapses,fs)
		json.dump(synapsesWeights,fw)
		print("Donee saving=====")

	def shouldStopTraining(self,pm,asse,nextSession=False):

		session=self.session+1 if nextSession else self.session
		return pm<=ReSuMe.PERFORMANCE_THRESHOLD or self.shouldStopNow or session>=ReSuMe.TRAINING_THRESHOLD or asse<=ReSuMe.filter_tau

	#ReSuMe implementation here
	def train(self,desiredSignals):

		self.session 		= 0
		self.desiredSignals = desiredSignals

		spikesDetectorIdx 	= 0
		pms 				= []
		pm 					= float("inf")
		asse 				= float("inf")
		minPm 				= float("inf")
		minAsse				= float("inf")

		while True:
			 
			print("Session nr="+str(self.session)+"=============================================\n\n\n")
			desiredSignalsIdx=0
			incommingSynapses,synapsesWeights=[],[]
			nest.Simulate(self.cycle)

			pm 		= self.computePerformanceIndex(desiredSignals)
			asse 	= self.computeAverageSpikeShiftError(desiredSignals)
			
			if pm<minPm:
				minPm=pm

			if asse<minAsse:
				minAsse=asse
			
			pms.append(pm)

			print("Performance Index",pm,"Best Performance Index",minPm)
			print("Average Spike Shift Error",asse,"Best Average Spike Shift Error",minAsse)

			if self.shouldStopTraining(pm,asse):

				self.plotResults(pms,asses)
				self.saveWeights()
				break

			self.updateSpikeDetection(True)

			while  desiredSignalsIdx<len(desiredSignals)  or spikesDetectorIdx<len(self.spikeTimes) :
					
				if  desiredSignalsIdx<len(desiredSignals) and ( spikesDetectorIdx>=len(self.spikeTimes) or desiredSignals[desiredSignalsIdx][1]<self.spikeTimes[spikesDetectorIdx]):
					
					self.updateSynapticStrengths(desiredSignalsIdx,spikesDetectorIdx,True,desiredSignals)
					desiredSignalsIdx+=1
				
				else:
					
					if( self.spikeSenders[spikesDetectorIdx]>=self.learningNeurons[0] 
					and self.spikeSenders[spikesDetectorIdx]<=self.learningNeurons[len(self.learningNeurons)-1]):

						self.updateSynapticStrengths(desiredSignalsIdx,spikesDetectorIdx,False,[])
					
					spikesDetectorIdx+=1	

			if self.plotNow:
				self.plotResults(pms,asses)
				self.plotNow=False

			self.resetFunction(self.resetParams[0],self.resetParams[1])
			desiredSignals=[[sig[0],sig[1]+self.cycle] for sig in desiredSignals]

			self.session+=1

	def learningWindow(self,timeDifference,excitatory,teacher):

		A 	= ReSuMe.lw_a 	if teacher else -ReSuMe.lw_a
		#if not excitatory:  A 	= -A
		tau = ReSuMe.tau_d 	if teacher else  ReSuMe.tau_l

		return A*np.exp(-timeDifference/tau)

	def getTargetInputNeurons(self,delay,time,index):

		ind=index
		target=[]
		while ind-1>=0 and ind-1<len(self.spikeTimes) and (time-self.spikeTimes[ind-1])<=delay and (time-self.spikeTimes[ind-1])>=0 :
			if( self.spikeSenders[ind-1]>=self.learningPreSynNeurons[0] 
				and self.spikeSenders[ind-1]<=self.learningPreSynNeurons[len(self.learningPreSynNeurons)-1]):
				target.append(self.spikeSenders[ind-1])
			ind-=1
		if ind!=index:
			return 1,target 
		else:
			return -1,[]

	def updateIndependantChange(self,neuronId,time,desiredSignals):

		inputs 		= self.learningPreSynNeurons
		inhInput 	= self.inhPreSynNeurons
		excInput 	= np.delete(inputs,inhInput,0).tolist()
		incommingInhSynapses,inhSynapsesWeights = self.getIncommingWeights(neuronId,inhInput,False)
		incommingExcSynapses,excSynapsesWeights = self.getIncommingWeights(neuronId,excInput,False)

		ad 	= ReSuMe.independantChange 	#for desired spikes
		al 	=-ReSuMe.independantChange	#for real postsynaptic spikes

		inputTrain,learnTrain=self.getSpikeTrains(time)
		deltaInh = self.codeSignalsIntoTrain(desiredSignals,time)[neuronId]*(-ad)+learnTrain[neuronId]*(-al)
		deltaExc = self.codeSignalsIntoTrain(desiredSignals,time)[neuronId]*(ad)+learnTrain[neuronId]*(al)

		inhW = [{"weight":x+deltaInh} for x in inhSynapsesWeights]
		excW = [{"weight":x+deltaExc} for x in excSynapsesWeights]

		nest.SetStatus(incommingInhSynapses,inhW)
		nest.SetStatus(incommingExcSynapses,excW)


	def updateSynapticStrengths(self,desiredSignalsIdx,spikesDetectorIdx,teacher,desiredSignals):

		neuronId 	= desiredSignals[desiredSignalsIdx][0] if teacher else self.spikeSenders[spikesDetectorIdx]
		time 		= desiredSignals[desiredSignalsIdx][1] if teacher else self.spikeTimes[spikesDetectorIdx]
		error,target= self.getTargetInputNeurons(ReSuMe.filter_tau,time,spikesDetectorIdx)
		#self.updateIndependantChange(neuronId,time,desiredSignals)

		if error!=-1 and len(target)>0:

			incommingSynapses,synapsesWeights=self.getIncommingWeights(neuronId,target,False)

			ad 	= ReSuMe.independantChange 	# for desired spikes
			al 	=-ReSuMe.independantChange	# for real postsynaptic spikes

			for idx, val in enumerate(incommingSynapses):

				excitatory=synapsesWeights[idx]>=0 
				#if not excitatory: 
				#	ad  = -ad  # d=desired == teacher ad constant
				#	al 	= -al  # l=learner  al  constant
				timeDifference=0 
				sumWD = 0 # integration part of equation (4) 
				sumWL = 0 # integration part of equation (4) 
				while timeDifference<=ReSuMe.filter_tau*5: #time difference in the learning window / notation taken from original paper
					  
					  inputTrain,learnTrain=self.getSpikeTrains(time-timeDifference,ReSuMe.dt)
					  sumWD+=self.learningWindow(timeDifference,excitatory,True)*inputTrain[val[0]]
					  sumWL+=self.learningWindow(timeDifference,excitatory,False)*inputTrain[val[0]]
					  timeDifference+=ReSuMe.dt

				inputTrain,learnTrain=self.getSpikeTrains(time)
				delta_w=self.codeSignalsIntoTrain(desiredSignals,time)[neuronId]*(ad+sumWD)+learnTrain[neuronId]*(al+sumWL)
				print(delta_w,synapsesWeights[idx],val[0],val[1])
				if excitatory:#synapsesWeights[idx]>=0:
					synapsesWeights[idx]=max(0.00000000001,synapsesWeights[idx]+delta_w)
				else: 
					#if synapsesWeights[idx]<0:
					synapsesWeights[idx]=min(-0.00000000001,synapsesWeights[idx]+delta_w)
					#else:
						#synapsesWeights[idx]+=delta_w
		
			self.setWeigths(incommingSynapses,synapsesWeights)

	def setWeigths(self,incommingSynapses,synapsesWeights):

		W=[{"weight":x} for x in synapsesWeights]
		nest.SetStatus(incommingSynapses,W)

	def getIncommingWeights(self,neurons,targetInputNeurons,all,targetOnly=False):
		
		neurons=np.array([neurons])
		neurons=neurons.flatten()
		if not all and not targetOnly:
			incommingSynapses	= nest.GetConnections(targetInputNeurons,target=neurons.tolist())
			synapsesWeights		= nest.GetStatus(incommingSynapses,'weight')
			synapsesWeights		= np.asarray(synapsesWeights)
			return incommingSynapses,synapsesWeights
		elif targetOnly:
			incommingSynapses	= nest.GetConnections(target=neurons.tolist())
			synapsesWeights		= nest.GetStatus(incommingSynapses,'weight')
			synapsesWeights		= np.asarray(synapsesWeights)
			return incommingSynapses,synapsesWeights
		else:
			incommingSynapses	= nest.GetConnections()
			synapsesWeights		= nest.GetStatus(incommingSynapses,'weight')
			synapsesWeights		= np.asarray(synapsesWeights)
			return incommingSynapses,synapsesWeights

def shouldAddToSpikeTrain(time1,time2,integrationDelay=0):

	return  time2-time1<=integrationDelay+1 and time1<=time2