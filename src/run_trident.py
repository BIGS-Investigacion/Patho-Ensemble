import argparse
import sys
from run_batch_of_slides import main

parser = argparse.ArgumentParser(description="Batch of slides")
parser.add_argument('--task', type=str)
parser.add_argument('--wsi_dir', type=str)
parser.add_argument('--job_dir', type=str)
parser.add_argument('--patch_encoder', type=str)
parser.add_argument('--patch_size', type=str)
parser.add_argument('--mag', type=str)
args = parser.parse_args()

sys.argv = [
    "run_batch_of_slides",
    '--task', args.task, \
    '--wsi_dir', args.wsi_dir, \
    '--job_dir', args.job_dir, \
    '--patch_encoder', args.patch_encoder, \
    '--patch_size', args.patch_size, \
    '--mag', args.mag, \
]

main()
