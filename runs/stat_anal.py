import argparse
import os
import pandas as pd

from lib.utils import make_unless_exits, print_argparse

def get_corruptions(fpth:str, idx_nm:str) -> list[str]:
    report = pd.read_csv(fpth)
    return report[idx_nm].unique().tolist()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--report_list', type=str)
    ap.add_argument('--output_path', type=str)
    ap.add_argument('--output_file_name', type=str)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--arch', type=str)
    ap.add_argument('--index_name', type=str, default='Corruption')
    ap.add_argument('--bef_adpt', type=str, default='Befor Adaptation')
    ap.add_argument('--aft_adpt', type=str, default='After Adaptation')
    args = ap.parse_args()
    args.report_list = [it.strip() for it in args.report_list.split(',')]
    make_unless_exits(args.output_path)
    print_argparse(args)
    ##########################################
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Corruption', 'Before-mean', 'Before-std', 'After-mean', 'After-std', 'Improve-mean', 'Improve-std'])
    for i, report_addr in enumerate(args.report_list):
        report = pd.read_csv(report_addr)
        if i == 0: reports = [report]
        else: reports.append(report)
    unity_report = pd.concat(reports, ignore_index=True)
    for corruption in get_corruptions(fpth=args.report_list[0], idx_nm=args.index_name):
        sub_report = unity_report[unity_report[args.index_name]==corruption]
        B_mean = round(sub_report[args.bef_adpt].mean(), ndigits=4)
        B_std = round(sub_report[args.bef_adpt].std(), ndigits=4)
        A_mean = round(sub_report[args.aft_adpt].mean(), ndigits=4)
        A_std = round(sub_report[args.aft_adpt].std(), ndigits=4)
        I_mean = round((sub_report[args.aft_adpt]-sub_report[args.bef_adpt]).mean(), ndigits=4)
        I_std = round((sub_report[args.aft_adpt]-sub_report[args.bef_adpt]).std(), ndigits=4)
        records.loc[len(records)] = [args.dataset, args.arch, corruption, B_mean, B_std, A_mean, A_std, I_mean, I_std]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))
    print('END!')