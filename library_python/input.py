import argparse
import json


def input_rydbergs(json_file=''):

    if json_file != '':
        print(f'Loading input from {json}')
        with open(json_file) as j:
            args = json.load(j)
            return args
            
    parser = argparse.ArgumentParser(description='Parameters Hamiltonian spin')

    # Hamiltonian parameters
    parser.add_argument('--L', type = int, default = 10, help = 'Number of sites in the system')
    parser.add_argument('--V1', type = float , default = 1., help = 'density-density interaction 1')
    parser.add_argument('--V2', type = float , default = 2., help = 'density-density interaction 2')
    parser.add_argument('--Omega', type = float , default = -0.05, help = 'Rabi frequency')
    parser.add_argument('--Omega2', type = float , default = -0.05, help = 'Rabi frequency of additional drive field')
    parser.add_argument('--eps', type = float , default = 0., help = 'drive field detuning with antiblockade resonance')

    # initial state properties
    parser.add_argument('--Z', type = int , default = 1, help = 'Number of excitations')
    parser.add_argument('--index_Z', type = int , default = 0 , help = 'Label of sampled product state')

    # time of simulation
    parser.add_argument('--T' , type = float , default = 10, help = 'Time of simulation in units of Omega')
    parser.add_argument('--number_steps' , type = int , default = 100, help = 'number of steps')

    # fluctuations
    parser.add_argument('--sigma'  , nargs='+' , default = [0.,0.001,0.001], help = 'standard deviation along three directions')
    parser.add_argument('--index_seed', type = int , default = 0 , help = 'seed for random number generator')


    args = vars(parser.parse_args())

    return args



def input_DMRGX(json_file=''):

    if json_file != '':
        print(f'Loading input from {json}')
        with open(json_file) as j:
            args = json.load(j)
            return args
            
    parser = argparse.ArgumentParser(description='Parameters Hamiltonian spin East-West')

    # Hamiltonian parameters
    parser.add_argument('--L', type = int, default = 2, help = 'Number of sites in the system')
    parser.add_argument('--s', type = float , default = 1., help = 'Kinetic constraint')

    # Initial state parameters
    parser.add_argument("--initial_state", type = str, default = 'kink', help="Initial state")
    parser.add_argument('--n', type = int, default = 2, help = 'Number of excitations')

    # DMRG-X parameters
    parser.add_argument('--dmrgx_maxD', type = int, default = 20, help = 'max bond dimension in DMRG-X')
    parser.add_argument('--dmrgx_tol', type = float , default = 1E-11, help = 'tolerance of DMRG-X')


    args = vars(parser.parse_args())

    return args

# def input_H_east_west_spin(json_file=''):

#     if json_file != '':
#         print(f'Loading input from {json}')
#         with open(json_file) as j:
#             args = json.load(j)
#             return args
            
#     parser = argparse.ArgumentParser(description='Parameters Hamiltonian spin East-West')

#     parser.add_argument('--L', type = int, default = 2, help = 'Number of sites in the system')
#     parser.add_argument('--s', type = float , default = 1., help = 'Kinetic constraint')

#     ########################################################################################

#     args = vars(parser.parse_args())

#     return args

# def input_initial_state(json_file=''):

#     if json_file != '':
#         print(f'Loading input from {json}')
#         with open(json_file) as j:
#             args = json.load(j)
#             return args
            
#     parser = argparse.ArgumentParser(description='Initial state')
    
#     ########################################################################################

#     args = vars(parser.parse_args())

#     return args