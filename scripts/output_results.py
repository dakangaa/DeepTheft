import torch
import pandas as pd
import numpy as np
import subprocess
import re
import argparse


def test(HyperParameters, Origin_domain_nums, test_domain, args):

    path = "results/MateModel_Hyper"
    if args.mode == "O":
        var2 = Origin_domain_nums
        indexes = pd.MultiIndex.from_product(
            [HyperParameters, Origin_domain_nums],
            names=["HyperParameters", "Origin_domain_nums"]
        )
        const_var = test_domain[0]
    elif args.mode == "T":
        var2 = test_domain
        indexes = pd.MultiIndex.from_product(
            [HyperParameters, test_domain],
            names=["HyperParameters", "test_domain"]
        )
        const_var = Origin_domain_nums[0]

    columns = ["VAL_ACC", "VAL_F1", "TEST_ACC", "TEST_F1"]
    df = pd.DataFrame(np.zeros((len(HyperParameters) * len(var2), 4)), index=indexes, columns=columns)

    for hp in HyperParameters:
        for v in var2:
            log = "HyperParameter:{}\t var2:{}\t \nloading checkpoint..."
            print(log.format(hp, v))
            print("testing...")
            if args.mode == "O":
                file = path + "/" + hp + "_" + str(v) + "_" + const_var + "_train_ckpt.pth"
                test_cmd = ["python", "Step2_Layer-wise_Hyperparameter_Inferring/test.py",
                            "-H", hp, "-o", str(v), "--device", args.device, "--test_domain", args.test_domain, "--workers", "3"]
            elif args.mode == "T":
                file = path + "/" + hp + "_" + str(const_var) + "_" + v + "_train_ckpt.pth"
                test_cmd = ["python", "Step2_Layer-wise_Hyperparameter_Inferring/test.py",
                            "-H", hp, "-o", str(const_var), "--device", args.device, "--test_domain", v, "--workers", "3"]
            test_result = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in test_result.stdout.split("\n"):
                if line.startswith("TEST"):
                    df.loc[(hp, v), "TEST_F1"] = float(re.search(r"F1:([0-9.]+)", line).group(1))
                    df.loc[(hp, v), "TEST_ACC"] = float(re.search(r"Acc:([0-9.]+)", line).group(1))
            checkpoint = torch.load(file)
            df.loc[(hp, v), "VAL_ACC"] = checkpoint["acc"][0]
            df.loc[(hp, v), "VAL_F1"] = checkpoint["f1"][0]

            print(f"[HP:{hp} \t|var2:{v}] \t|ACC:{df.loc[(hp, v), 'TEST_ACC']} \t|F1:{df.loc[(hp, v), 'TEST_F1']}")

    return df

def read_epoch(HyperParameters, Origin_domain_nums, columns):
    path = "results/MateModel_Hyper"
    indexes = pd.MultiIndex.from_product(
        [HyperParameters, Origin_domain_nums],
        names=["HyperParameters", "Origin_domain_nums"]
    )
    df = pd.DataFrame(np.zeros((len(HyperParameters) * len(Origin_domain_nums), len(columns))), index=indexes, columns=columns)
    for hp in HyperParameters:
        for od in Origin_domain_nums:
            log = "HyperParameter:{}\t Origin_domain_nums:{}\t \nloading checkpoint..."
            print(log.format(hp, od))
            file = path + "/" + hp + "_" + str(od) + "_train_ckpt.pth"
            checkpoint = torch.load(file, map_location=torch.device('cpu'))
            for col in columns:
                if col not in checkpoint.keys():
                    df.loc[(hp, od), col] = float("nan")
                    continue
                if col in ["acc", "f1", "loss_value"]:
                    df.loc[(hp, od), col] = checkpoint[col][0]
                else:
                    df.loc[(hp, od), col] = checkpoint[col]
    return df

if __name__ == "__main__":
    # read epoch
    # HyperParameters = ["kernel_size", "out_channels", "stride"]
    # Origin_domain_nums = [1,2,3,4]
    # columns = ["epoch", "acc", "f1", "loss_value"]
    # df = read_epoch(HyperParameters, Origin_domain_nums, columns)
    # print(df)

    # test
    parser = argparse.ArgumentParser(description='collect data')
    parser.add_argument("--device", type=str, default="autodl", help="laptop or autodl")
    parser.add_argument("--mode", type=str, default="O", help="T est_domain or O rigin_domain_nums")
    args = parser.parse_args()

    HyperParameters = ["kernel_size", "out_channels", "stride"]
    if args.mode == "O":
        origin_domain_nums = [1,2,3,4]
        test_domain = ["331"]
        df = test(HyperParameters, origin_domain_nums, test_domain, args)
        print(df)
        with pd.ExcelWriter("results/results.xlsx", if_sheet_exists="replace", mode="a") as writer:
            df.to_excel(writer, sheet_name="origin_domain_num")
    elif args.mode == "T":
        origin_domain_nums = [4]
        test_domain = ["160", "192", "224", "299", "331"]
        df = test(HyperParameters, origin_domain_nums, test_domain, args)
        print(df)
        with pd.ExcelWriter("results/results.xlsx", if_sheet_exists="replace", mode="a") as writer:
            df.to_excel(writer, sheet_name="test_domain")
    else:
        raise ValueError
