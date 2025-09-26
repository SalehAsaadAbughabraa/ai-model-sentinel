
import os
import subprocess
import eval

def execute_command(cmd):
    """Execute system command"""
    return subprocess.call(cmd, shell=True)

def evaluate_code(code_string):
    """Evaluate code string"""
    return eval(code_string)

# Dangerous patterns
os.system("echo 'This is dangerous'")