# encoding: utf-8
import os
gpu = 0
trials = [1,2,3,4,5,6,7,8,9,10]
for i in trials:
    file_data = ""
    f = open("train_regdb.sh")
    lines = f.readlines()
    with open("train_regdb.sh", "w") as fw:
        for line in lines:
            print(line)
            if "gpu" in line:
                line = "--gpu " + str(gpu) + " \\" + '\n'
            if "trial" in line:
                line = "--trial " + str(i) + " \\" + '\n'
            file_data += line
        fw.write(file_data)
    os.system('sh train_regdb.sh')
