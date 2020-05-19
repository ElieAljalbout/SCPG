import numpy as np 
import time
import serial

# configure the serial connections (the parameters differs on the device you are connecting to)


def sendSerial(ser,commands,duration):

	#ser.write(command)
	commands=list(set(commands))
	command=','.join([str(i) for i in commands])
	command= '<'+command+','+str(int(duration))+'>'
	
	if ser.isOpen():

		ser.write(command)
		time.sleep((duration+500.)/1000)


def startSerial(port):

	ser = serial.Serial(
    port=port,
    baudrate=19200
	)

	if not ser.isOpen():
		
		try: 
		    ser.open()
		except Exception, e:
		    print "error opening serial port: " + str(e)
		    exit()

	return ser

def lagProcessing(ts,evs,maxT,minpt,port):

	ser = startSerial(port)
	time.sleep(5)
	commands 	= []
	lastVal 	= 0
	evs= [i-minpt for i in evs]
	for idx,val in enumerate(ts):

		if len(commands)>0 and abs(val-lastVal) >=maxT:

			sendSerial(ser,commands,val-lastVal)
			commands 	= []
			lastVal 	= val

		if len(commands)==0: 
			lastVal = val

		commands.append(evs[idx])

	ser.close()


def lagProcessingBatch(ts,evs,maxT,minpt,port):

	ser = startSerial(port)
	time.sleep(5)
	commands 	= []
	cmd         = '<'
	lastVal 	= 0
	evs= [i-minpt for i in evs]
	for idx,val in enumerate(ts):

		if len(commands)>0 and abs(val-lastVal) >=maxT:

			commands=list(set(commands))
			command=','.join([str(i) for i in commands])
			if(len(cmd)>1):cmd=cmd+'-'
			cmd =cmd+ command+','+str(int(val-lastVal))
			commands 	= []
			lastVal 	= val

		if len(commands)==0: 
			lastVal = val

		commands.append(evs[idx])
	cmd=cmd+'>'
	ser.write(cmd)
	ser.close()




