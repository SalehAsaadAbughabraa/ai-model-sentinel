import os
import subprocess

def dangerous_operation():
    os.system("echo 'dangerous'")
    subprocess.call(["ls", "-la"])
    eval("print('eval test')")
    return "This is dangerous code"

dangerous_operation()