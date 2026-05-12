import matplotlib.pyplot as plt
import streamlit as st

from ..constants import CLASS_COLORS, TARGET


def page_overview(df):
    st.title("Dataset Overview")

    total_missing = int(df.isnull().sum().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features", str(df.shape[1] - 1))
    c3.metric("Disease Classes", str(df[TARGET].nunique()))
    c4.metric("Cells with Missing Data", f"{total_missing:,}")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Class Distribution")
        counts = df[TARGET].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [CLASS_COLORS.get(c, "#3498db") for c in counts.index]
        bars = ax.barh(counts.index, counts.values, color=colors)
        for bar, val in zip(bars, counts.values):
            pct = val / len(df) * 100
            ax.text(
                bar.get_width() + 80, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)", va="center", fontsize=9,
            )
        ax.set_xlabel("Count")
        ax.set_xlim(0, counts.max() * 1.35)
        ax.set_title("Liver Disease Class Distribution (Original Data)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with right:
        st.subheader("Missing Values")
        missing_pct = df.isnull().mean() * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        if missing_pct.empty:
            st.info("No missing values in dataset.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(missing_pct.index, missing_pct.values, color="#e74c3c")
            for i, v in enumerate(missing_pct.values):
                ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
            ax.set_xlabel("Missing (%)")
            ax.set_xlim(0, missing_pct.max() * 1.35)
            ax.set_title("Columns with Missing Values")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.subheader("Data Sample (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)
