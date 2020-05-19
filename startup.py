from neural_phase_generator.NPG import *
from CPG import *
import common.Processing as pr
from TonicDrive import *
import pylab
import operator 
import argparse


def createPreNetNodes(demo,tdFrequency=1/4.0,tdNeuronsNr=1):

	spikedetector 	= nest.Create("spike_detector",params={"withtime": True,"withgid": True})
	starterNode 	= nest.Create("spike_generator")
	spikeTimes 		= []
	if demo:
		spikeTimes      = np.arange(0.1,1020.0*10.0,4.0)
		spikeTimes 		= np.concatenate([spikeTimes,np.arange(spikeTimes[-1]+2,spikeTimes[-1]+700.0*10.0,2.0)])
	
	tonicDrive 		= TonicDrive(tdFrequency,tdNeuronsNr,spikeTimes=spikeTimes)
	tonicDrive.create()
	
	return tonicDrive,spikedetector,starterNode

def main(demo=False,visualize=False,train=[],load=[],freq=1/4.0):

	nest.SetKernelStatus({'local_num_threads': 8})
	tonicDrive,spikedetector,starterNode = createPreNetNodes(demo,tdFrequency=freq)
	npg=NPG()

	if demo:
		load = ["weights/synapses-3-3-6","weights/Weights-3-3-6"]

	cpg=CPG(createPreNetNodes
			,tonicDrive
			,spikedetector
			,npg
			,motorNeuronsNr=24
			,trainModel=len(train)>0
			,desiredSignals=train
			#,desiredSignals=[[3,82],[9,106],[1,108],[8,120],[5,135],[7,190],[5,297],[4,340],[3,383],[0,415],[2,601],[7,654],[9,687],[6,730]]
			#,desiredSignals=[[14,20],[20,25],[0,122],[8,124],[12,220],[18,224],[7,326],[1,328],[13,424],[19,426],[17,521],[23,525],[3,620],[11,624],[15,724],[21,726],[4,816],[10,831],[16,920],[22,925]]
			,loadWeights=len(load)>0
			,path=load#["weights/synapses-3-3-6","weights/Weights-3-3-6"]
			,starterNode=starterNode)

	cpg.buildNetwork()

	nest.Simulate(18000)
	dSD = nest.GetStatus(spikedetector,keys="events")[0]

	evs = dSD["senders"]
	ts = dSD["times"]

	if len(ts)>0 :ts, evs = zip(*sorted(zip(ts, evs),key=operator.itemgetter(0), reverse=False))
	#comb=np.vstack((ts, evs)).T

	if demo:
		pr.lagProcessingBatch(ts,evs,20,cpg.motorNeurons[0],'/dev/ttyACM0')
	
	if visualize:
		
		print(ts)
		#print(comb)
		pylab.figure(1)
		pylab.plot(ts, evs, ".")
		pylab.xlabel('Time (ms)')
		pylab.ylabel('Neuron GID (NEST)')
		pylab.show()


if __name__ == "__main__": 
	'''
    usage: main.py [-h] [--demo] [--vis] [-t T] [-l L] [-f F]
        
                            
    optional arguments:
	  -h, --help  show this help message and exit
	  --demo      Show the demo
	  --vis       Visualize the output spikes
	  -t T        train with the given vector of desired spikes in comma separated
	              format e.g: [[1,102,3],[5,206].....,[NID,time]]
	  -l L        load the synapses and weights from the paths: 'SPATH,WPATH'
	  -f F        Frequency of the CPG's input (spike/ms)
    '''


	parser = argparse.ArgumentParser(description='Spiking CPG')
	parser.add_argument("--demo", action='store_true', help="Show the demo")
	parser.add_argument("--vis", action='store_true', help="Visualize the output spikes")
	parser.add_argument("-t", help="train with the given vector of desired spikes in comma separated format e.g: [[1,102,3],[5,206].....,[NID,time]]")
	parser.add_argument("-l", help="load the synapses and weights from the paths: 'SPATH,WPATH'")
	parser.add_argument("-f", type=float, help="Frequency of the CPG's input (spike/ms)")
	args = parser.parse_args()

	# Train/Visualize/Demo/Set_param as per the arguments
	load 	= []
	train 	= []
	freq 	= 1/4.0

	if args.l:
		load 	= args.l.split(',')
	if args.f:
		freq 	= args.f
	if args.t:
		traint	= [int(i) for i in args.t.split(',')]
		temp 	= [0,0]
		train	= []
		for idx,val in enumerate(traint):
			if idx%2==0:
				temp[0]=val
			else:
				temp[1]=val
				train.append(temp)

	main(args.demo,args.vis,load=load,freq=freq,train=train)