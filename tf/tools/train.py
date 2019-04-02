import sys
sys.path.append('./')
from optparse import OptionParser
from colorizer.tf.solver import Solver
from colorizer.tf.solver_multigpu import SolverMultigpu
from colorizer.tf.utils import process_config

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",  
                  help="configure filename")
(options, args) = parser.parse_args() 
if options.configure:
  conf_file = str(options.configure)
else:
  print('please specify --conf configure filename')
  exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)
if len(str(common_params['gpus']).split(','))==1:
  solver = Solver(True, common_params, solver_params, net_params, dataset_params)
else:
  solver = SolverMultigpu(True, common_params, solver_params, net_params, dataset_params)
solver.train_model()