 
rule CriticalCommands {
    strings:
        $rm = "rm -rf" nocase
        $format = "format c:" nocase
        $system = "os.system(" 
    condition:
        any of them
}

rule NetworkThreats {
    strings:
        $socket = "import socket"
        $connect = ".connect("
    condition:
        $socket and $connect
}

rule PickleThreats {
    strings:
        $reduce = "__reduce__"
        $pickle = "pickle."
    condition:
        $reduce and $pickle
}