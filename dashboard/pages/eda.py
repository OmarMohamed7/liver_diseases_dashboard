import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from ..constants import BINARY, BINARY_LABELS, NUMERICAL, TARGET


def page_eda(df):
    st.title("Exploratory Data Analysis")

    tab_num, tab_binary, tab_corr = st.tabs(
        ["Numerical Features", "Symptoms & Comorbidities", "Correlations"]
    )

    with tab_num:
        selected = st.selectbox("Select feature", NUMERICAL)
        col_hist, col_box = st.columns(2)

        with col_hist:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df[selected].dropna(), bins=40, color="#3498db", edgecolor="white", linewidth=0.4)
            ax.set_title(f"Distribution — {selected}")
            ax.set_xlabel(selected)
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_box:
            fig, ax = plt.subplots(figsize=(6, 4))
            classes = sorted(df[TARGET].unique())
            groups = [df[df[TARGET] == cls][selected].dropna() for cls in classes]
            labels = [c[:20] for c in classes]
            ax.boxplot(groups, labels=labels) # pyright: ignore[reportCallIssue]
            ax.set_title(f"{selected} by Disease Class")
            ax.set_ylabel(selected)
            plt.xticks(rotation=30, ha="right", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("Descriptive Statistics")
        st.dataframe(df[NUMERICAL].describe().round(2), use_container_width=True)

    with tab_binary:
        sym_cols = [c for c in BINARY if c.startswith("Sym_")]
        com_cols = [c for c in BINARY if c.startswith("Comorb_")]

        st.subheader("Symptom Prevalence")
        prevalence = {BINARY_LABELS[c]: df[c].mean() * 100 for c in sym_cols}
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(prevalence.keys(), prevalence.values(), color="#e67e22") # pyright: ignore[reportArgumentType]
        for bar, v in zip(bars, prevalence.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_ylabel("Prevalence (%)")
        ax.set_ylim(0, max(prevalence.values()) * 1.2)
        ax.set_title("Symptom Prevalence across the Dataset")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Comorbidity Prevalence")
        prev_c = {BINARY_LABELS[c]: df[c].mean() * 100 for c in com_cols}
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(prev_c.keys(), prev_c.values(), color="#9b59b6") # pyright: ignore[reportArgumentType]
        for bar, v in zip(bars, prev_c.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_ylabel("Prevalence (%)")
        ax.set_ylim(0, max(prev_c.values()) * 1.2)
        ax.set_title("Comorbidity Prevalence across the Dataset")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab_corr:
        st.subheader("Pearson Correlation — Numerical Features")
        corr = df[NUMERICAL].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", ax=ax,
            cmap="coolwarm", center=0, linewidths=0.5, annot_kws={"size": 8},
        )
        ax.set_title("Numerical Feature Correlations")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
