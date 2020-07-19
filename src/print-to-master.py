import sys
from src.pretty_print import  print_for_master_thesis_compact, print_for_master_thesis

print_folder = f'{sys.argv[1]}/*/'
print_for_master_thesis(print_folder, ['features', 'layer'])
print_for_master_thesis_compact(print_folder, ['features', 'layer'])
