import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--airpls", type=bool, default=True, help="number of image channels")
parser.add_argument("--SG", type=bool, default=True, help="number of image channels")
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--l2", type=float, default=0.0002, help="adam: learning rate")
