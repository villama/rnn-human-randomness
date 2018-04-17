# config.py ---

import argparse


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for data
data_arg = add_argument_group("Data")

data_arg.add_argument("--num_moves", type=int,
                      default=1200,
                      help="Number of moves per person in data")

data_arg.add_argument("--input_dtype", type=str,
                      default="csv",
                      help="Datatype of the input: either db or csv")

data_arg.add_argument("--input", type=str,
                      default="usermoves.csv",
                      help="SQLite3 file with user data")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--batch_size", type=int,
                       default=1,
                       help="Size of each training batch")

train_arg.add_argument("--num_epoch", type=int,
                       default=20,
                       help="Number of epochs to train")

train_arg.add_argument("--log_dir", type=str,
                       default="logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--test_size", type=int,
                       default=400,
                       help="Number of samples set aside for training")

train_arg.add_argument("--n", type=int,
                       default=None,
                       help="Number of participants to study")

# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--num_neurons", type=int,
                       default=32,
                       help="Number of neurons in the hidden layer")

model_arg.add_argument("--num_hidden", type=int,
                       default=1,
                       help="Number of hidden layers")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()
