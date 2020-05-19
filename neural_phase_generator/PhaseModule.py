import nest

# This class represents the generator of neurons and relative synapses 
# responsible for the creation of one phase module in the NPG
# Author: Elie Aljalbout

IAF_TYPE	= "iaf_psc_alpha"

class PhaseModule:

	phaseCounter = 0
	starterSet   = False
	linkDuration = 35.0 # approximatly every connection between 2 interneurons or and h/t neuron with an interneuron has this duration


	def __init__(self,durationApprox=180.0,starter=False,name=None):

		if starter and PhaseModule.starterSet: print("WARNING","Multiple starters set for NPG" )

		self.name 			= name if name else "phase"+str(PhaseModule.phaseCounter+1)
		self.durationApprox = durationApprox # max duration the module can be active for
		self.neurons 		= []
		self.starter 		= starter

		PhaseModule.phaseCounter+=1

	# Create module's neurons
	def createModuleNeurons(self):

		self.interNr 		= self.durationApprox//PhaseModule.linkDuration-1
		self.hNeuron 		= nest.Create(IAF_TYPE) # Neuron respresenting the acitivity of the module
		self.interNeurons 	= nest.Create(IAF_TYPE,int(self.interNr))
		self.tNeuron 		= nest.Create(IAF_TYPE) # Transition neuron, to move from phase to phase
		self.qNeuron 		= nest.Create(IAF_TYPE) # Neuron inhibiting other modules to assure only one phase is active in one period

		self.neurons.append(self.hNeuron[0])
		self.neurons.extend(list(self.interNeurons))
		self.neurons.append(self.tNeuron[0])
		self.neurons.append(self.qNeuron[0])

		#if self.starter: 
		#	nest.SetStatus(self.hNeuron, {"V_m": -54.0})
		#	PhaseModule.starterSet=True

	# Connect module's neurons
	def connectModuleNeurons(self):


		nest.Connect(self.neurons[:len(self.neurons)-2], self.neurons[:len(self.neurons)-2],"one_to_one", {"weight":1100.0})#,"delay":10.0})
		nest.Connect(self.neurons[:len(self.neurons)-2], self.qNeuron,"all_to_all", {"weight": 2000})

		for i in range(len(self.neurons)-2):
			nest.Connect([self.neurons[i]],[self.neurons[i+1]],syn_spec={"weight":250.0,"delay":20.0})
			nest.Connect([self.neurons[i+1]],[self.neurons[i]],syn_spec={"weight":-2500.0})

		nest.Connect(self.tNeuron, self.qNeuron, syn_spec={"weight":-4000})
		nest.Connect(self.tNeuron, self.neurons[:-2],"all_to_all", {"weight":-4000})

	# Establish connection from phase to phase (transition)
	def connectToNextModule(self,module):

		nest.Connect(self.tNeuron, module.hNeuron, syn_spec={"weight": 20000}) 

	def inhibitModules(self,modules):

		for module in modules:
			nest.Connect(self.qNeuron, module.neurons[:-1], "all_to_all", {"weight":-2000})

	def inhibitModulesPopulation(self, population):

		nest.Connect(self.tNeuron, population,"all_to_all",{"weight":-8000})


	# connect the modules corresponding neurons to a given spike detector
	# connectTo corresponds to : connect to h, connect to q, connect to t
	def connectToSpikeDetector(self,spikeDetector,connectTo=[True,False,False,False]): 

		if connectTo[0]: nest.Connect(self.hNeuron,spikeDetector)
		if connectTo[1]: nest.Connect(self.neurons[1:len(self.neurons)-2],spikeDetector)
		if connectTo[2]: nest.Connect(self.qNeuron,spikeDetector)
		if connectTo[3]: nest.Connect(self.tNeuron,spikeDetector)


	def create(self):

		self.createModuleNeurons()
		self.connectModuleNeurons()