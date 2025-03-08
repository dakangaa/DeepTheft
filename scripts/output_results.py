import torch
import pandas as pd
import numpy as np
import subprocess
import re
import argparse


def test(HyperParameters, Origin_domain_nums, args):

    path = "results/MateModel_Hyper"
    indexes = pd.MultiIndex.from_product(
        [HyperParameters, Origin_domain_nums],
        names=["HyperParameters", "Origin_domain_nums"]
    )
    columns = ["VAL_ACC", "VAL_F1", "TEST_ACC", "TEST_F1"]
    df = pd.DataFrame(np.zeros((len(HyperParameters) * len(Origin_domain_nums), 4)), index=indexes, columns=columns)

    for hp in HyperParameters:
        for od in Origin_domain_nums:
            log = "HyperParameter:{}\t Origin_domain_nums:{}\t \nloading checkpoint..."
            print(log.format(hp, od))
            file = path + "/" + hp + "_" + str(od) + "_train_ckpt.pth"
            checkpoint = torch.load(file)
            df.loc[(hp, od), "VAL_ACC"] = checkpoint["acc"][0]
            df.loc[(hp, od), "VAL_F1"] = checkpoint["f1"][0]

            print("testing...")
            test_cmd = ["python", "Step2_Layer-wise_Hyperparameter_Inferring/test.py", 
                        "-H", hp, "-o", str(od), "--device", args.device, "--test_domain", args.test_domain]
            test_result = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in test_result.stdout.split("\n"):
                if line.startswith("TEST"):
                    df.loc[(hp, od), "TEST_F1"] = float(re.search(r"F1:([0-9.]+)", line).group(1))
                    df.loc[(hp, od), "TEST_ACC"] = float(re.search(r"Acc:([0-9.]+)", line).group(1))
            print(f"[HP:{hp} \t|OD:{od}] \t|ACC:{df.loc[(hp, od), 'TEST_ACC']} \t|F1:{df.loc[(hp, od), 'TEST_F1']}")

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
    HyperParameters = ["kernel_size", "out_channels", "stride"]
    Origin_domain_nums = [1,2,3,4]
    columns = ["epoch", "acc", "f1", "loss_value"]
    df = read_epoch(HyperParameters, Origin_domain_nums, columns)
    print(df)

    # test
    # parser = argparse.ArgumentParser(description='collect data')
    # parser.add_argument("--device", type=str, default="autodl", help="laptop or autodl")
    # parser.add_argument("--test_domain", type=str, default="331", help="测试域")
    # args = parser.parse_args()
    # HyperParameters = ["kernel_size", "out_channels", "stride"]
    # Origin_domain_nums = [1,2,3,4]
    # df = test(HyperParameters, Origin_domain_nums, args)
    # print(df)

    with pd.ExcelWriter("results/results.xlsx", if_sheet_exists="replace", mode="a") as writer:
        df.to_excel(writer, sheet_name="append")