# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests



# ==========================================================
# 1. 路径设置
# ==========================================================


BASE_DIR = Path(
    r"E:\Multi-omic Immunity\GCN_immune\scripts\LGG\picture\figure_4\figure_4_C_fuji\new_20260903"
)


EDGE_FILE = (
    BASE_DIR /
    "overall_edge.xlsx"
)


CLUSTER_FILE = (
    BASE_DIR /
    "consensus_patient_assignments.xlsx"
)


OUTPUT_DIR = (
    BASE_DIR /
    "1vsrest_BH_all_methods_results"
)



# ==========================================================
# 参数
# ==========================================================


ALPHA_NORMALITY = 0.05

ALPHA_VARIANCE = 0.05

ALPHA_OMNIBUS = 0.05


PATIENT_PREFIX = "P_"


EDGE_KEYS = [
    "src",
    "dst"
]



GROUP_NAMES = [
    "group_0",
    "group_1",
    "group_2",
    "group_3"
]



# ==========================================================
# 数据清洗
# ==========================================================


def clean_numeric(values):

    x = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).to_numpy(
        dtype=float
    )


    return x[
        np.isfinite(x)
    ]



# ==========================================================
# Shapiro-Wilk
# ==========================================================


def shapiro_pvalue(x):

    if len(x)<3:
        return np.nan


    if len(x)>5000:

        rng=np.random.default_rng(2026)

        x=rng.choice(
            x,
            size=5000,
            replace=False
        )


    return float(
        stats.shapiro(x).pvalue
    )



# ==========================================================
# Welch ANOVA
# ==========================================================


def welch_anova(groups):


    arrays=list(groups.values())


    n=np.array(
        [
            len(x)
            for x in arrays
        ],
        dtype=float
    )


    means=np.array(
        [
            np.mean(x)
            for x in arrays
        ]
    )


    vars=np.array(
        [
            np.var(
                x,
                ddof=1
            )
            for x in arrays
        ]
    )


    weights=n/vars


    mean_weight=np.sum(
        weights*means
    )/np.sum(weights)



    correction=np.sum(
        (
            1-
            weights/weights.sum()
        )**2 /
        (n-1)
    )


    df1=len(arrays)-1


    df2=(
        len(arrays)**2-1
    )/(
        3*correction
    )


    numerator=(
        np.sum(
            weights*
            (means-mean_weight)**2
        )
        /
        df1
    )


    denominator=(
        1+
        (
            2*(len(arrays)-2)
            /
            (len(arrays)**2-1)
        )
        *
        correction
    )


    F=numerator/denominator


    p=stats.f.sf(
        F,
        df1,
        df2
    )


    return F,p,df1,df2



# ==========================================================
# 两组比较
# ==========================================================


def one_vs_rest_tests(groups):


    results=[]

    raw_p=[]


    for focal in groups:


        x1=groups[focal]


        rest=[
            g for g in groups
            if g!=focal
        ]


        x2=np.concatenate(
            [
                groups[g]
                for g in rest
            ]
        )


        p1=shapiro_pvalue(x1)

        p2=shapiro_pvalue(x2)


        normal1=(
            p1>=ALPHA_NORMALITY
            if np.isfinite(p1)
            else False
        )

        normal2=(
            p2>=ALPHA_NORMALITY
            if np.isfinite(p2)
            else False
        )



        if normal1 and normal2:


            levene=stats.levene(
                x1,
                x2,
                center="median"
            )


            equal_var=(
                levene.pvalue
                >=
                ALPHA_VARIANCE
            )


            if equal_var:


                test="Student t-test"


                res=stats.ttest_ind(
                    x1,
                    x2,
                    equal_var=True
                )


            else:

                test="Welch t-test"


                res=stats.ttest_ind(
                    x1,
                    x2,
                    equal_var=False
                )


            statistic=res.statistic

            p=res.pvalue


        else:


            test="Mann-Whitney U"


            res=stats.mannwhitneyu(
                x1,
                x2
            )


            statistic=res.statistic

            p=res.pvalue



        raw_p.append(p)



        results.append(
            {
                "focal_group":focal,
                "test":test,
                "statistic":statistic,
                "p_raw":p,
                "mean_focal":np.mean(x1),
                "mean_rest":np.mean(x2)
            }
        )


    # BH

    q=multipletests(
        raw_p,
        method="fdr_bh"
    )[1]


    for r,v in zip(results,q):

        r["p_adj"]=v

        r["significant"]=(
            v<ALPHA_OMNIBUS
        )


    return results



# ==========================================================
# 核心统计
# ==========================================================


def run_statistics(
        group_dfs,
        output_folder
):


    indexed={}

    patient_cols={}


    for name,df in zip(
        GROUP_NAMES,
        group_dfs
    ):


        cols=[
            c for c in df.columns
            if c.startswith(PATIENT_PREFIX)
        ]


        patient_cols[name]=cols


        indexed[name]=df.set_index(
            EDGE_KEYS
        )

    # ==========================================================
    # find the edge
    # ==========================================================

    GENE_COLUMNS = [
        "src_gene",
        "dst_gene"
    ]

    # 只取当前表中实际存在的节点名称列
    annotation_cols = [
        c for c in GENE_COLUMNS
        if c in group_dfs[0].columns
    ]

    # 先找到所有组共同存在的 edge
    common_edges = (
        group_dfs[0][EDGE_KEYS]
        .drop_duplicates()
        .copy()
    )

    for df in group_dfs[1:]:
        common_edges = common_edges.merge(
            df[EDGE_KEYS].drop_duplicates(),
            on=EDGE_KEYS,
            how="inner"
        )

    # 从第一个 group 中取得 src_gene / dst_gene
    annotations = (
        group_dfs[0][
            EDGE_KEYS + annotation_cols
            ]
        .drop_duplicates(
            subset=EDGE_KEYS
        )
    )


    common_edges = common_edges.merge(
        annotations,
        on=EDGE_KEYS,
        how="left"
    )



    overall=[]

    one_rest=[]



    for _,edge in common_edges.iterrows():


        key=(
            edge["src"],
            edge["dst"]
        )


        groups={}


        for name in GROUP_NAMES:


            row=indexed[name].loc[key]


            cols=patient_cols[name]


            groups[name]=clean_numeric(
                row[cols]
            )



        normality={

            g:shapiro_pvalue(x)

            for g,x in groups.items()

        }



        all_normal=all(
            p>=ALPHA_NORMALITY
            for p in normality.values()
        )



        if not all_normal:


            test="Kruskal-Wallis"


            res=stats.kruskal(
                *groups.values()
            )


            p=res.pvalue



        else:


            levene=stats.levene(
                *groups.values(),
                center="median"
            )


            if levene.pvalue < ALPHA_VARIANCE:


                test="Welch ANOVA"


                _,p,_,_=welch_anova(
                    groups
                )

            else:


                test="ANOVA"


                res=stats.f_oneway(
                    *groups.values()
                )


                p=res.pvalue

        overall_row = {
            "src": edge["src"],
            "src_gene": edge.get("src_gene", np.nan),
            "dst": edge["dst"],
            "dst_gene": edge.get("dst_gene", np.nan),
            "test": test,
            "p": p
        }

        overall.append(overall_row)



        if p<ALPHA_OMNIBUS:


            result=one_vs_rest_tests(
                groups
            )

            for r in result:
                r["src"] = edge["src"]

                r["src_gene"] = edge.get(
                    "src_gene",
                    np.nan
                )

                r["dst"] = edge["dst"]

                r["dst_gene"] = edge.get(
                    "dst_gene",
                    np.nan
                )

                one_rest.append(r)



    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    pd.DataFrame(
        overall
    ).to_csv(
        output_folder /
        "edge_overall_tests.csv",
        index=False
    )


    pd.DataFrame(
        one_rest
    ).to_csv(
        output_folder /
        "edge_one_vs_rest_tests.csv",
        index=False
    )



# ==========================================================
# main
# ==========================================================


def main():



    edge_df=pd.read_excel(
        EDGE_FILE,
        sheet_name="Sheet1"
    )


    cluster_df=pd.read_excel(
        CLUSTER_FILE
    )



    methods=[
        c for c in cluster_df.columns
        if c!="Patient_ID"
    ]



    patients=[
        c for c in edge_df.columns
        if c.startswith(PATIENT_PREFIX)
    ]



    edge_info=[
        c for c in edge_df.columns
        if c not in patients
    ]



    print(
        "发现方法:",
        len(methods)
    )



    for method in methods:


        print(
            "\n运行:",
            method
        )



        groups=[]



        clusters=sorted(
            cluster_df[method]
            .dropna()
            .unique()
        )



        for cluster in clusters:



            ids=(
                cluster_df
                .loc[
                    cluster_df[method]==cluster,
                    "Patient_ID"
                ]
                .astype(str)
                .tolist()
            )



            ids=[
                i for i in ids
                if i in patients
            ]



            df=edge_df[
                edge_info+
                ids
            ].copy()



            # row_mean

            df["row_mean"]=(
                df[ids]
                .mean(axis=1)
            )


            groups.append(df)



        result_path=(
            OUTPUT_DIR /
            method
        )


        run_statistics(
            groups,
            result_path
        )



    print("\nall complete")



if __name__=="__main__":

    main()