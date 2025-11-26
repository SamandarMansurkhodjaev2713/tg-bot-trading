import subprocess
import sys

def run(cmd: list[str]):
    print('>',' '.join(cmd))
    subprocess.check_call(cmd)

def main():
    run([sys.executable, 'scripts/etl_external.py'])
    run([sys.executable, 'scripts/preprocess_external.py'])
    run([sys.executable, 'scripts/train_external_pipeline.py'])
    print('Educational pipeline executed successfully.')

if __name__ == '__main__':
    main()

