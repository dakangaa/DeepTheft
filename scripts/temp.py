import torch
import pandas as pd
import numpy as np
import subprocess
import re
from itertools import product

def test(HyperParameters, Origin_domain_nums):

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
                        "-H", hp, "-o", str(od)]
            test_result = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in test_result.stdout.split("\n"):
                if line.startswith("TEST"):
                    df.loc[(hp, od), "TEST_F1"] = float(re.search(r"F1:([0-9.]+)", line).group(1))
                    df.loc[(hp, od), "TEST_ACC"] = float(re.search(r"Acc:([0-9.]+)", line).group(1))

    print("result:")
    print(df)
    return df

def read_epoch(HyperParameters, Origin_domain_nums):
    path = "results/MateModel_Hyper"
    indexes = pd.MultiIndex.from_product(
        [HyperParameters, Origin_domain_nums],
        names=["HyperParameters", "Origin_domain_nums"]
    )
    columns = ["EPOCH"]
    df = pd.DataFrame(np.zeros((len(HyperParameters) * len(Origin_domain_nums), 1)), index=indexes, columns=columns)
    for hp in HyperParameters:
        for od in Origin_domain_nums:
            log = "HyperParameter:{}\t Origin_domain_nums:{}\t \nloading checkpoint..."
            print(log.format(hp, od))
            file = path + "/" + hp + "_" + str(od) + "_train_ckpt.pth"
            checkpoint = torch.load(file)
            df.loc[(hp, od), "EPOCH"] = checkpoint["epoch"]
    return df

if __name__ == "__main__":
    # read epoch
    HyperParameters = ["kernel_size", "out_channels", "stride"]
    Origin_domain_nums = [1,2,3]
    df = read_epoch(HyperParameters, Origin_domain_nums)
    print(df)
    with pd.ExcelWriter("results/results.xlsx", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="epoch")

    # test
    # HyperParameters = ["kernel_size", "out_channels", "stride"]
    # Origin_domain_nums = [1,2,3]
    # df = test(HyperParameters, Origin_domain_nums)