from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import csv
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from TA-NIDS.env import make_eval_env
from TA-NIDS.ppo2 import PPO2, evaluate_policy
from TA-NIDS.utils import generate_csv_writer

def argsparser():
    parser = argparse.ArgumentParser("Active Anomaly Detection")
    parser.add_argument('--test', help='Testing datasets', default='NBtest')
    parser.add_argument('--budget', help='Budget in testing', type=int, default=10000)
    parser.add_argument('--log', help='the directory to save the evaluation results', default='resultsnsl')
    parser.add_argument('--load', help='the model directory', default='log/model0001.zip')#log/model_2000_ids2017_1.zip
    parser.add_argument('--eval_interval', help='the interval of recording results in evaluation', type=int, default=10)

    return parser
    
def evaluate(args):

    if not os.path.exists(args.log):
        os.makedirs(args.log)

    test_datasets = args.test.split(',')

    # Make the testing environments
    eval_envs = {}
    for d in test_datasets:
        path = os.path.join('./data', d+'.csv')
        output_path = os.path.join(args.log, d+'.csv') 
        csv_file, csv_writer = generate_csv_writer(output_path)
        eval_envs[d] = {'env': make_eval_env(datapath=path, budget=args.budget),
                        'csv_writer': csv_writer,
                        'csv_file': csv_file,
                        'mean_reward': 0,
                       }

    # Load model
    model = PPO2.load(args.load)
    wocnmd = {}
    for d in eval_envs:
        print('Dataset {}'.format(d))
        print('-----------------------')
        wocrr, reward, _, results, wocresults, thex, they = evaluate_policy(model, eval_envs[d]['env'], n_eval_episodes=1, deterministic=False, use_batch=True)
        print('Reward: {}'.format(reward))
        wocnmd[d] = str(reward) + "#"  + str(wocrr)

        eval_envs[d]['csv_writer'].writerow(results)
        eval_envs[d]['csv_writer'].writerow(wocresults)
        eval_envs[d]['csv_writer'].writerow(thex)
        eval_envs[d]['csv_writer'].writerow(they)
        eval_envs[d]['csv_file'].flush()
        eval_envs[d]['csv_file'].close()
        print('Results written!')
        print('')
    print(wocnmd)
    print(args.load)
if __name__ == "__main__":
    from pudb import set_trace

    set_trace()

    parser = argsparser()
    args = parser.parse_args()
    evaluate(args)

