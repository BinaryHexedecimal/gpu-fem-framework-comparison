import argparse
import sys
import json
import subprocess

from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
target_folder = project_root / "data/raw_data/"
target_folder.mkdir(parents=True, exist_ok=True)


meshes_lst = ["coarse", "m1", "m2", "fine"] 
#meshes_lst = ["coarse"] 
models_lst = ["gravity", "traction", "compression"]
versions_lst = ["B", "G"]
framework = "warp"



def warp_in_batch(numerical_runtimes, real_runtimes, precompute_runtimes, 
                  dtype_name, auto_force_input, save):

    
    numerical_res_lst = []
    real_res_cpu_lst = []
    real_res_gpu_lst = []
    real_res_zero_lst = []

    for mesh in meshes_lst:
        print(f"Processing mesh: {mesh}")
        for model in models_lst:
            print(f"Processing model: {model}")
            for version in versions_lst:
                    print(f"Processing version: {version}")

                    # ------- numerical runtime ------- #
                    cmd_numerical = [
                        "python",
                        "run_one_case_numerical.py",
                        "--mesh", mesh,
                        "--model", model,
                        "--version", version,
                        "--dtype_name", dtype_name,
                        "--numerical_runtimes", str(numerical_runtimes),
                    ]
                    res_numerical = subprocess.run(
                        cmd_numerical,
                        capture_output=True,
                        text=True
                    )
                    if res_numerical.returncode != 0 or res_numerical.stderr.strip():
                        print("STDERR repr:")
                        print(repr(res_numerical.stderr))
                    # print("STDOUT repr:")
                    # print(repr(res_numerical.stdout))

                    stdout_lines = res_numerical.stdout.strip().splitlines()
                    json_str = stdout_lines[-1]
                    data_numerical = json.loads(json_str)
                    numerical_res_lst.append(data_numerical)


                    # ------- real runtime on CPU ------- #
                    cmd_real_cpu = [
                        "python",
                        "run_one_case_realtime_cpu.py",
                        "--mesh", mesh,
                        "--model", model,
                        "--version", version,
                        "--dtype_name", dtype_name,
                        "--real_runtimes", str(real_runtimes),
                        "--precompute_runtimes", str(precompute_runtimes),
                    ]
                    if auto_force_input:
                        cmd_real_cpu.append("--auto_force_input")
                    res_real_cpu = subprocess.run(
                        cmd_real_cpu,
                        capture_output=True,
                        text=True
                    )

                    if res_real_cpu.returncode != 0 or res_real_cpu.stderr.strip():
                        print("STDERR:")
                        print(res_real_cpu.stderr)
                    
                    # print("STDOUT:")
                    # print(res_real_cpu.stdout)

                    stdout_lines = res_real_cpu.stdout.strip().splitlines()
                    json_str = stdout_lines[-1]
                    data_real_cpu = json.loads(json_str)
                    real_res_cpu_lst.append(data_real_cpu)




                    #"""
                    
                    # ------- real runtime on GPU ------- #
                    cmd_real_gpu = [
                        "python",
                        "run_one_case_realtime_gpu.py",
                        "--mesh", mesh,
                        "--model", model,
                        "--version", version,
                        "--dtype_name", dtype_name,
                        "--real_runtimes", str(real_runtimes),
                        "--precompute_runtimes", str(precompute_runtimes),
                    ]
                    if auto_force_input:
                        cmd_real_gpu.append("--auto_force_input")
                    
                    res_real_gpu = subprocess.run(
                        cmd_real_gpu,
                        capture_output=True,
                        text=True
                    )

                    if res_real_gpu.returncode != 0 or res_real_gpu.stderr.strip():
                        print("STDERR:")
                        print(res_real_gpu.stderr)
                    # print("STDOUT:")
                    # print(res_real_gpu.stdout)

                    stdout_lines = res_real_gpu.stdout.strip().splitlines()
                    json_str = stdout_lines[-1]
                    data_real_gpu = json.loads(json_str)
                    real_res_gpu_lst.append(data_real_gpu)


                    # ------- real runtime on zero copy ------- #
                    cmd_real_zero = [
                        "python",
                        "run_one_case_realtime_zero.py",
                        "--mesh", mesh,
                        "--model", model,
                        "--version", version,
                        "--dtype_name", dtype_name,
                        "--real_runtimes", str(real_runtimes),
                        "--precompute_runtimes", str(precompute_runtimes),
                    ]
                    if auto_force_input:
                        cmd_real_zero.append("--auto_force_input")
                    res_real_zero = subprocess.run(
                        cmd_real_zero,
                        capture_output=True,
                        text=True
                    )
                    if res_real_zero.returncode != 0 or res_real_zero.stderr.strip():
                        print("STDERR:")
                        print(res_real_zero.stderr)

                    # print("STDOUT:")
                    # print(res_real_zero.stdout)

                    stdout_lines = res_real_zero.stdout.strip().splitlines()
                    json_str = stdout_lines[-1]
                    data_real_zero = json.loads(json_str)
                    real_res_zero_lst.append(data_real_zero)

                    #"""                    

    if save:

        filename_numerical = f"{framework}_{dtype_name}_numerical_res.json"
        numerical_file_path = target_folder / filename_numerical
        with open(numerical_file_path, "w") as f:
            json.dump(numerical_res_lst, f, indent=4)

        filename_real = f"{framework}_{dtype_name}_realtime_cpu_res.json"
        real_file_path = target_folder / filename_real
        with open(real_file_path, "w") as f:
            json.dump(real_res_cpu_lst, f, indent=4)

        filename_real_gpu = f"{framework}_{dtype_name}_realtime_gpu_res.json"
        real_gpu_file_path = target_folder / filename_real_gpu
        with open(real_gpu_file_path, "w") as f:
            json.dump(real_res_gpu_lst, f, indent=4)

        filename_real_zero = f"{framework}_{dtype_name}_realtime_zero_res.json"
        real_zero_file_path = target_folder / filename_real_zero
        with open(real_zero_file_path, "w") as f:
            json.dump(real_res_zero_lst, f, indent=4)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp",
        type=int,
        default=64,
        choices=[32, 64],
        help="Use float64 precision by default"
    )


    parser.add_argument(
        "--real_runtimes",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--precompute_runtimes",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--numerical_runtimes",
        type=int,
        default=4,
    )
    
    # with action="store_true" the behavior is:
    # absent -> False
    # present -> True
    parser.add_argument(
        "--save",
        action="store_true"
    )

    parser.add_argument(
        "--auto_force_input",
        action="store_true"
    )
    
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()

    numerical_runtimes = args.numerical_runtimes
    real_runtimes = args.real_runtimes
    precompute_runtimes = args.precompute_runtimes

    save = args.save
    dtype_name = "fp64" if args.fp == 64 else "fp32"

    auto_force_input = args.auto_force_input

    warp_in_batch(numerical_runtimes, real_runtimes, precompute_runtimes, 
                  dtype_name, auto_force_input, save)