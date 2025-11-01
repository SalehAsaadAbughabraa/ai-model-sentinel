class SafeEvaluator:
    def safe_eval(self, expr):
        allowed={'sin':__import__('math').sin,'cos':__import__('math').cos,'sqrt':__import__('math').sqrt,'abs':abs,'min':min,'max':max,'len':len,'sum':sum,'round':round}
        return eval(expr,{'__builtins__':{}},allowed)
    def safe_exec(self, code):
        restricted_globals={'__builtins__':{'range':range,'len':len,'str':str,'int':int,'float':float,'list':list,'dict':dict}}
        exec(code, restricted_globals, {})
        return restricted_globals
