import os
import base64
import socket

# أنماط مشبوهة متعمدة
encoded_cmd = base64.b64encode(b"cmd.exe /c dir")
eval("print('simulated malicious eval')")
exec("print('simulated exec')")
os.system("echo 'test system call'")

# اتصال شبكي simulado
try:
    s = socket.socket()
    s.connect(("8.8.8.8", 80))  # Google DNS for test
    s.close()
except:
    pass

# أنماط AI خطرة محتملة
import pickle
import numpy as np

class MaliciousModel:
    def __reduce__(self):
        return (os.system, ('echo "malicious"',))

# محاولة pickle خطرة
model = MaliciousModel()
"""

with open("test_malicious.py", "w") as f:
    f.write(malicious_code)

print("✅ تم إنشاء ملف اختبار مشبوه")