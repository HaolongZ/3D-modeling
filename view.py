
import sys
from subprocess import call

def main():
    pred_file_name = sys.argv[1]
    call(['meshlab', pred_file_name])

if __name__ == '__main__':

    main()
