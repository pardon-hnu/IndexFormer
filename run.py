from execute.tpch_time_ratio import tpch_time_ratio_run
from execute.tpch_time_cost import tpch_time_cost_run
from execute.tpcds_time_ratio import tpcds_time_ratio_run
from execute.tpcds_time_cost import tpcds_time_cost_run
from execute.imdb_time_ratio import imdb_time_ratio_run
from execute.imdb_time_cost import imdb_time_cost_run

dataset='tpch'
prediction_object='cost'    # ratio or cost
toe='Training'              # Training or Evaluation
pt_file=''                  #file path of best checkpoint for evaluation

if dataset == 'tpch':
    if prediction_object == 'ratio':
        tpch_time_ratio_run(toe,pt_file)
    elif prediction_object == 'cost':
        tpch_time_cost_run(toe,pt_file)
    else:
        raise ValueError('error object')
elif dataset == 'tpcds':
    if prediction_object == 'ratio':
        tpcds_time_ratio_run(toe,pt_file)
    elif prediction_object == 'cost':
        tpcds_time_cost_run(toe,pt_file)
    else:
        raise ValueError('error object')
elif dataset == 'imdb':
    if prediction_object == 'ratio':
        imdb_time_ratio_run(toe,pt_file)
    elif prediction_object == 'cost':
        imdb_time_cost_run(toe,pt_file)
    else:
        raise ValueError('error object')
else:
    raise ValueError('error dataset')