
# this file contains all the constants used in the neuron simulations
# Author : Elie Aljalbout

IAF_TYPE	= "iaf_psc_alpha"

w_min_inh =-100
w_max_inh =-20

w_min_exc = 1
w_max_exc = 2

delay_min = 1
delay_max = 20

C_m_bound=[100.0,300.0]
tau_m_bound=[9.0,30.0]
E_L_bound=[-90.0,-70.0]
V_th_bound=[-50.0,-30]
V_reset_bound=[-90.0,-60.0]
I_e_bound=[0.0,150.0]

params={"C_m":C_m_bound,
		"tau_m":tau_m_bound,
		"E_L":E_L_bound,
		"V_th":V_th_bound,
		"V_reset":V_reset_bound,
		"I_e":I_e_bound}

seed=[123+i for i in range(len(params))]

rnSeeds= [5975+i for i in range(50)]
numpyGSeed=666777

