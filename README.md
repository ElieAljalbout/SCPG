# Task-Independent Spiking Central Pattern Generator: A Learning-Based Approach #

#### in NEPL 2020 [[Project]](https://sites.google.com/view/task-independent-cpg/home) [[PDF]](https://link.springer.com/content/pdf/10.1007/s11063-020-10224-9.pdf)

If you use this implementation, please cite our paper.
```
@article{aljalbout2020task,
  title={Task-Independent Spiking Central Pattern Generator: A Learning-Based Approach},
  author={Aljalbout, Elie and Walter, Florian and R{\"o}hrbein, Florian and Knoll, Alois},
  journal={Neural Processing Letters},
  pages={1--14},
  year={2020},
  publisher={Springer}
}
```

#### Requirements
- NEST 
- numpy
- pylab

Usage
--------
Use the main script for training, visualizing spikes or for real implementation
```
python startup.py <options>
```
Option     | |
-------- | ---
```-h ```| `` Help``
```--demo ```| `` To start the demo with the ALLBOT robot``
```--vis ```| ``Visualize the output spikes of the motor neurons``
``-t Train`` | ``Trains the CPG for the behavior specified in train such that '0,123,2,156,3,288,1,523' will be transformed into [[0,123],[2,156],[3,288],[1,523]] where [0,123] refers to the zeroth motor neuron and 123 is its desired spiking time``
``-l Load``| `` Load the CPG model which is saved in the paths specified in Load such that 'synapses,weights' will be transformed into ['synapses','weights'] and synapses is the file where the synapse specification is saved, and weights contains the corresponding weights ``


Project Structure
------------------------
Folder / File     | Description|
-------- | ---
<i class="icon-folder-open"></i> weights| Contains the files where the CPG models are saved
<i class="icon-folder-open"></i> neural_phase_generator | Contains the code for the neural phase generator class and relative ones
<i class="icon-folder-open"></i> common| Contains common functionality codes and constants
<i class="icon-file"></i> TonicDrive.py | Contains the tonic drive, created with artificial spike generator
<i class="icon-file"></i> startup.py | The main python script for training and using the CPG
<i class="icon-file"></i> ReSuMe.py | Contains the implementation of the remote supervision method, note: this implementation is the general one.
<i class="icon-file"></i>CPG.py| Contains the class of the CPG builder

### Important notes
- The default CPG has four modules (each representing a phase in the locomotion), and the modules has default phase duration [200,300,200,300]
- To change this update the npg construction call in startup.py in the main function: npg(modulesNr=N,phasesDuration=[t1,t2,....])

### Not learning?
- Possible that the cycle time is not detected exactly, and this might require a time shift in every duration, you can manually set it in the function 'updateNetworkModel' at line: 'cycleTime=...' , or fix the method? It is working almost all the time, and only failed once during the tests (Due to time limitation we didn't invest in fixing it since it's only for luxury)
- The learning window (filter_tau) you're using is too small or the pattern forming network size is small, to figure it out, print and plot the ouput of the pattern forming network and compare the spikes time to the ones desired in the motor neurons, for instance if you want a motor neuron to spike at t=213ms and the closest spike time in the pattern forming network is at 206ms you might decide to use a learning window equal to : 213-206
 
