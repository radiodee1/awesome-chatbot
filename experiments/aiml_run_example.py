#!/usr/bin/python3

import aiml
import sys

if not sys.warnoptions:
    import warnings, os
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

# Create the kernel and learn AIML files
kernel = aiml.Kernel()
kernel.verbose(False)
kernel.learn("std_startup.xml")
#kernel.respond("load aiml b")

# Press CTRL-C to break this loop
while True:
    r = kernel.respond(input("> "))
    print(r)
