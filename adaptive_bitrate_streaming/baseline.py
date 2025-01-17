import run_baseline
import argparse
import sys, os
import time  # 用于计算运行时间

model_list = [
    "genet",
    "mpc",
    "bba",
]

test_list = [
    "norway",
    "5g",
    "fcc",
    "fcc-test",
    "ghent",
    "oboe",
]

vedio_list = [
    "video1",
    # "video2",
]

if __name__ == "__main__":
    for video in vedio_list:
        for model in model_list:
            for test in test_list:
                print(
                    f"python run_baseline.py --model {model} --test-trace {test} --video {video} >> output.log"
                )
                start_time = time.time()
                os.system(
                    f"python run_baseline.py --model {model} --test-trace {test} --video {video} --test-trace-num {-1} >> output.log"
                )
                # 记录结束时间
                elapsed_time = time.time() - start_time
                # 运行时间写入 time.log
                with open("time.log", "a") as time_log:
                    time_log.write(f"Model: {model}_dataset {test}\n")
                    time_log.write(f"Time Taken: {elapsed_time:.2f} seconds\n\n")
