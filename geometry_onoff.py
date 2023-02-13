import re
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('file_path')
parser.add_argument('onoff')
args = parser.parse_args()

print('\nfile path is: ' + args.file_path)
print('\nonoff = ' + args.onoff)

file = open(args.file_path + ".in", 'r')
lines = file.readlines()
file.close()

if args.onoff == 'on' and lines[-1][:2] == '##':
    lines[-1] = lines[-1][1:]
    print('\ngeometry view is ON!\n')
if args.onoff == 'off' and lines[-1][:2] == '#g':
    lines[-1] = '#' + lines[-1]
    print('\ngeometry view is OFF!\n')

file = open(args.file_path + ".in", 'w')
file.writelines(lines)
file.close()
