from recbole.quick_start import run_recbole
import pdb
pdb.set_trace()
config_file_list = ['CMGAN-ml.yaml']
run_recbole(model='CMGAN', dataset='ml-1m', config_file_list=config_file_list)

