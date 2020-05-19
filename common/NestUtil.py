import nest
import Constants
import json
import array
import numpy as np

BATCHSIZE=1

def setPopParams(pop):
	
	for idx,val in enumerate(Constants.params):
		np.random.seed(Constants.seed[idx])
		dprop =  [{val: Constants.params[val][0]+(Constants.params[val][1]-Constants.params[val][0])*np.random.rand()} for x in pop]
		nest.SetStatus(pop,dprop)

def loadNetWeights(synapsesFilePath,weightFilePath):

	synapsesJsonData = open(synapsesFilePath).read()
	weightsJsonData  = open(weightFilePath).read()
	   
	synapses = json.loads(synapsesJsonData)
	weights  = json.loads(weightsJsonData)

	weights  = [{"weight":x} for idx,x in enumerate(weights) ]
	synapses = [array.array('l',[val[0],val[1],val[2],val[3],val[4]]) for val in synapses ]

	#print(synapses[:10])
	synapses = tuple(synapses) # setStatus expects the synapses to be in a tuple
	nest.SetStatus(synapses,weights)

