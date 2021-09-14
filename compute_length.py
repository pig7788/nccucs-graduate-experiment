import sys

import utility

if __name__ == '__main__':
    path = sys.argv[1]
    length = int(sys.argv[2])
    print(utility.comput_all_input_sequence_length(path, length))
