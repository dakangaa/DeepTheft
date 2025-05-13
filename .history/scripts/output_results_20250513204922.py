import torch
import pandas as pd
import numpy as np
import subprocess
import re
import argparse


def test(layer_type, HyperParameters, Origin_domain_nums, test_domain, args):

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

    columns = ["VAL_ACC", "VAL_F1", "TEST_ACC", "TEST_F1", "TEST_R", "TEST_P"]
    df = pd.DataFrame(np.zeros((len(HyperParameters) * len(var2), 6)), index=indexes, columns=columns)

    for hp in HyperParameters:
        for v in var2:
            log = "HyperParameter:{}\t var2:{}\t \nloading checkpoint..."
            print(log.format(hp, v))
            print("testing...")
            if args.mode == "O":
                file = path + "/" + layer_type +"_"+ hp + "_" + str(v) + "_" + const_var + "_train_ckpt.pth"
                test_cmd = ["python", "Step2_Layer-wise_Hyperparameter_Inferring/test.py", "--layer_type", layer_type,
                            "-H", hp, "-o", str(v), "--device", args.device, "--test_domain", const_var, "--workers", "3",
                            "--path", args.path]
            elif args.mode == "T":
                file = path + "/" + layer_type +"_"+ hp + "_" + str(const_var) + "_" + v + "_train_ckpt.pth"
                test_cmd = ["python", "Step2_Layer-wise_Hyperparameter_Inferring/test.py", "--layer_type", layer_type,
                            "-H", hp, "-o", str(const_var), "--device", args.device, "--test_domain", v, "--workers", "3",
                            "--path", args.path]
            if args.regression and hp == "out_channels":
                test_cmd.append("--regression")
                file = file.replace("train", "regression")
            test_result = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(test_result.stdout)
            for line in test_result.stdout.split("\n"):
                if line.startswith("TEST"):
                    df.loc[(hp, v), "TEST_F1"] = float(re.search(r"F1:([0-9.]+)", line).group(1))
                    df.loc[(hp, v), "TEST_ACC"] = float(re.search(r"Acc:([0-9.]+)", line).group(1))
                    df.loc[(hp, v), "TEST_P"] = float(re.search(r"P:([0-9.]+)", line).group(1))
                    df.loc[(hp, v), "TEST_R"] = float(re.search(r"R:([0-9.]+)", line).group(1))

            checkpoint = torch.load(file)
            df.loc[(hp, v), "VAL_ACC"] = checkpoint["acc"][0]
            df.loc[(hp, v), "VAL_F1"] = checkpoint["f1"][0]

            print(f"[HP:{hp} \tvar2:{v}] \tACC:{df.loc[(hp, v), 'TEST_ACC']} \tF1:{df.loc[(hp, v), 'TEST_F1']}")

    return df


def read_ckpt(layer_type, hyperParameters, origin_domain_nums, test_domain, columns, is_regression=False):
    path = "results/MateModel_Hyper"

    indexes = pd.MultiIndex.from_product(
        [hyperParameters, origin_domain_nums],
        names=["HyperParameters", "Origin_domain_nums"]
    )
    df_od = pd.DataFrame(np.zeros((len(hyperParameters) * len(origin_domain_nums), len(columns))), index=indexes, columns=columns)
    indexes = pd.MultiIndex.from_product(
        [hyperParameters, test_domain],
        names=["HyperParameters", "test_domain"]
    )
    df_td = pd.DataFrame(np.zeros((len(hyperParameters) * len(test_domain), len(columns))), index=indexes, columns=columns)
    for hp in hyperParameters:
        for od in origin_domain_nums:
            log = "HyperParameter:{}\t Origin_domain_nums:{}\t \nloading checkpoint..."
            print(log.format(hp, od))
            if is_regression and hp == "out_channels":
                file = path + '/' + layer_type +"_"+ hp + "_" + str(od) + "_" + "331" + "_" + "regression" + '_ckpt.pth'
            else:
                file = path + '/' + layer_type +"_"+ hp + "_" + str(od) + "_" + "331" + "_" + "train" + '_ckpt.pth'
            checkpoint = torch.load(file, map_location=torch.device('cpu'))
            for col in columns:
                if col not in checkpoint.keys():
                    df_od.loc[(hp, od), col] = float("nan")
                    continue
                if col in ["acc", "f1", "loss_value"]:
                    df_od.loc[(hp, od), col] = checkpoint[col][0]
                else:
                    df_od.loc[(hp, od), col] = checkpoint[col]
        for td in test_domain:
            if is_regression and hp == "out_channels":
                file = path + '/' + layer_type +"_"+ hp + "_" + str(4) + "_" + td + "_" + "regression" + '_ckpt.pth'
            else:
                file = path + '/' + layer_type +"_"+ hp + "_" + str(4) + "_" + td + "_" + "train" + '_ckpt.pth'
            checkpoint = torch.load(file, map_location=torch.device('cpu'))
            for col in columns:
                if col not in checkpoint.keys():
                    df_td.loc[(hp, td), col] = float("nan")
                    continue
                if col in ["acc", "f1", "loss_value"]:
                    df_td.loc[(hp, td), col] = checkpoint[col][0]
                else:
                    df_td.loc[(hp, td), col] = checkpoint[col]
    return df_od, df_td

if __name__ == "__main__":
    # read epoch
    # HyperParameters = ["kernel_size", "out_channels", "stride"]
    # Origin_domain_nums = [1,2,3,4]
    # columns = ["epoch", "loss_value"]
    # test_domain = ["160", "192", "224", "299", "331"]
    # df_od, df_td = read_ckpt(HyperParameters, Origin_domain_nums, test_domain, columns)
    # print(df_od)
    # print(df_td)
    # with pd.ExcelWriter("results/results.xlsx", if_sheet_exists="replace", mode="a") as writer:
    #     df_od.to_excel(writer, sheet_name=args.layer_type + "O")
    # with pd.ExcelWriter("results/results.xlsx", if_sheet_exists="replace", mode="a") as writer:
    #     df_td.to_excel(writer, sheet_name=args.layer_type + "T")

    # test
    parser = argparse.ArgumentParser(description='collect data')
    parser.add_argument("--device", type=str, default="autodl", help="laptop or autodl")
    parser.add_argument("--mode", type=str, default="O", help="T est_domain or O rigin_domain_nums")
    parser.add_argument("--regression", action="store_true", help="out_channels预测是否为回归任务")
    parser.add_argument("--layer_type", type=str, default="conv2d")
    parser.add_argument('--path', default='results/MateModel_Hyper', type=str, help='load_path')
    args = parser.parse_args()

    if args.layer_type == "conv2d":
        # HyperParameters = ["kernel_size", "out_channels", "stride"]
        HyperParameters = ["kernel_size", "out_channels"] #TEST
    elif args.layer_type == "max_pool2d":
        HyperParameters = ["kernel_size", "padding"]
    elif args.layer_type == "linear":
        HyperParameters = ["out_channels"]

    if args.mode == "O":
        origin_domain_nums = [1,2,3,4]
        test_domain = ["331"]
        df = test(args.layer_type, HyperParameters, origin_domain_nums, test_domain, args)
        print(df)
        with pd.ExcelWriter("results/results.xlsx", if_sheet_exists="replace", mode="a") as writer:
            if args.regression:
                df.to_excel(writer, sheet_name=args.layer_type +"_"+ "O_regression")
            else:
                df.to_excel(writer, sheet_name=args.layer_type +"_"+ "O")
    elif args.mode == "T":
        origin_domain_nums = [4]
        test_domain = ["160", "192", "224", "299", "331"]
        df = test(args.layer_type, HyperParameters, origin_domain_nums, test_domain, args)
        print(df)
        with pd.ExcelWriter("results/results.xlsx", if_sheet_exists="replace", mode="a") as writer:
            if args.regression:
                df.to_excel(writer, sheet_name=args.layer_type +"_"+ "T_regression_RAPL1")
            else:
                df.to_excel(writer, sheet_name=args.layer_type +"_"+ "T_RAPL1")
    else:
        raise ValueError
