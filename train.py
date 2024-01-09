from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines import logger

from TA-NIDS.env import make_train_env, make_eval_env
from TA-NIDS.ppo2 import PPO2
from TA-NIDS.utils import generate_csv_writer

from pudb import set_trace
set_trace()

def argsparser():
    parser = argparse.ArgumentParser("Active Anomaly Detection")
    parser.add_argument('--train', help='Training datasets', default='0001')
    parser.add_argument('--test', help='Testing datasets', default='NBtest1')
    parser.add_argument('--budget', help='Budget in testing', type=int, default=100)
    parser.add_argument('--num_timesteps', help='The number of timesteps', type=int, default=20000000)
    parser.add_argument('--log', help='the directory to save logs and models', default='log')
    parser.add_argument('--eval_interval', help='the interval of recording results in evaluation', type=int, default=10)
    parser.add_argument('--rl_log_interval', help='the interval of RL log', type=int, default=10)
    parser.add_argument('--eval_log_interval', help='the interval of evaluation log on testing datasets', type=int, default=100)
    return parser
    
def train(args):

    anomaly_curve_log = os.path.join(args.log, 'anomaly_curves')
    if not os.path.exists(anomaly_curve_log):
        os.makedirs(anomaly_curve_log)

    train_datasets = args.train.split(',')
    test_datasets = args.test.split(',')

    # Generate the paths of datasets
    datapaths = []
    for d in train_datasets:
        datapaths.append(os.path.join('./data', d+'.csv'))

    # Make the training environment
    print("make train")
    env = make_train_env(datapaths)
    print("make test")
    # Make the testing environments
    eval_envs = {}
    for d in test_datasets:
        path = os.path.join('./data', d+'.csv')
        output_path = os.path.join(anomaly_curve_log, d+'.csv') 
        csv_file, csv_writer = generate_csv_writer(output_path)
        eval_envs[d] = {'env': make_eval_env(datapath=path, budget=args.budget),
                        'csv_writer': csv_writer,
                        'csv_file': csv_file,
                        'mean_reward': 0,
                       }

    # Train the model
    # Custom MLP policy of two layers of size 32 each with tanh activation function
    #policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[256, 256, 256])
    # Create the agent
    #model = PPO2("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.set_eval(eval_envs, args.eval_log_interval)
    model.learn(total_timesteps=args.num_timesteps, log_interval=args.rl_log_interval)
    model.save(os.path.join(args.log, 'model_test'))

if __name__ == "__main__":

    parser = argsparser()
    args = parser.parse_args()
    logger.configure(args.log)
    train(args)

