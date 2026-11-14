import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

CACHE = Path(__file__).parent / "online_retail.csv"

ACCENT = "#3B5BDB"
INK, MUTED, LINE, GRID = "#111827", "#6B7280", "#E4E7EC", "#EEF0F3"
# muted, distinguishable categorical set (segments)
SEG_COLORS = ["#3B5BDB", "#12A594", "#E5A00D", "#D6455D", "#7950F2", "#64748B", "#1C91D6", "#7A9A01"]

st.set_page_config(page_title="Customer Segmentation", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;}
.block-container {padding: 2rem 2.5rem 3rem; max-width: 1360px;}
header[data-testid="stHeader"] {background: transparent;}
section[data-testid="stSidebar"] {background:#0F172A; border-right:0;}
section[data-testid="stSidebar"] label p, section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"], section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"] {color:#CBD5E1 !important;}
section[data-testid="stSidebar"] [data-baseweb="select"] > div, section[data-testid="stSidebar"] [data-baseweb="input"] {background:#1E293B; border-color:#334155; color:#F1F5F9;}
section[data-testid="stSidebar"] [data-baseweb="select"] *, section[data-testid="stSidebar"] input {color:#F1F5F9 !important;}
section[data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}
h1,h2,h3 {letter-spacing:-0.01em;}

.brand {display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.05rem; color:#fff;}
.brand i {width:22px; height:22px; border-radius:6px; background:#3B5BDB; display:inline-block;
  box-shadow: inset -7px -7px 0 rgba(255,255,255,.28);}
.side-h {font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:#64748B; margin:22px 0 6px;}

.page-title {font-size:1.6rem; font-weight:700; color:#111827; margin:0;}
.page-sub {color:#6B7280; font-size:.92rem; margin:2px 0 18px;}

.strip {display:flex; background:#fff; border:1px solid #E4E7EC; border-radius:10px; margin-bottom:18px;}
.strip > div {flex:1; padding:16px 20px; border-right:1px solid #E4E7EC; min-width:0;}
.strip > div:last-child {border-right:0;}
.strip .l {font-size:.74rem; color:#6B7280; font-weight:500;}
.strip .v {font-size:1.55rem; font-weight:600; color:#111827; margin-top:4px; font-variant-numeric:tabular-nums;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.strip .s {font-size:.76rem; color:#6B7280; margin-top:2px;}

.pill {display:inline-block; background:#EEF2FF; color:#3B5BDB; border-radius:999px; padding:2px 10px; font-size:.74rem; font-weight:600; margin-right:6px;}
.up {color:#12805C; font-weight:600;} .down {color:#C2334B; font-weight:600;}
.strip svg {display:block; margin-top:8px;}
.finds {display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; margin-bottom:18px;}
.find {background:#fff; border:1px solid #E4E7EC; border-left:4px solid var(--c); border-radius:10px; padding:14px 18px;}
.find .k {font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:#6B7280;}
.find .t {font-size:.95rem; color:#111827; margin-top:4px; line-height:1.45;}
.find b {font-weight:650;}
.cards {display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; margin-bottom:18px;}
.card {background:#fff; border:1px solid #E4E7EC; border-top:4px solid var(--c); border-radius:10px; padding:14px 16px;}
.card .n {font-weight:600; color:#111827; font-size:.95rem;}
.card .big {font-size:1.7rem; font-weight:650; color:#111827; font-variant-numeric:tabular-nums; margin-top:2px;}
.card .big span {font-size:.78rem; color:#6B7280; font-weight:500; margin-left:4px;}
.card .bar {height:6px; background:#EEF0F3; border-radius:6px; margin:10px 0 4px; overflow:hidden;}
.card .bar i {display:block; height:100%; background:var(--c);}
.card .lbl {font-size:.74rem; color:#6B7280;}
.card .m {display:flex; justify-content:space-between; margin:10px 0; font-size:.78rem; color:#6B7280;}
.card .m b {display:block; color:#111827; font-size:.88rem;}
.card .act {font-size:.8rem; color:#374151; border-top:1px solid #EEF0F3; padding-top:9px;}
.h {font-size:.98rem; font-weight:600; color:#111827; margin:2px 0 0;}
.sub {font-size:.82rem; color:#6B7280; margin:0 0 6px;}

[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-color:#E4E7EC !important; border-radius:10px;}
button[data-baseweb="tab"] {font-weight:500; font-size:.95rem; padding: 10px 4px; margin-right: 18px;}
div[data-baseweb="tab-list"] {border-bottom:1px solid #E4E7EC; gap:0;}
.stButton>button, .stDownloadButton>button {border-radius:8px; font-weight:500;}

.phone {width:250px; margin:8px auto; border:1px solid #D0D5DD; border-radius:28px; padding:16px 12px 40px; background:#F6F7F9;}
.phone .bar {height:4px; width:48px; background:#D0D5DD; border-radius:4px; margin:0 auto 16px;}
.notif {background:#fff; border:1px solid #E4E7EC; border-radius:14px; padding:11px 13px;}
.notif .a {font-size:.68rem; color:#6B7280; font-weight:500;}
.notif .t {font-weight:600; margin:3px 0 2px; color:#111827; font-size:.9rem;}
.notif .b {font-size:.82rem; color:#374151; line-height:1.4;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
def base(fig, height=320, legend=False):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=INK), colorway=SEG_COLORS,
        margin=dict(l=4, r=4, t=8, b=4), showlegend=legend, title=None,
        legend=dict(orientation="h", y=-0.2, font=dict(size=11), title=None),
        hoverlabel=dict(bgcolor="#fff", bordercolor=LINE, font=dict(color=INK, size=12)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=11))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED), title_font=dict(color=MUTED, size=11))
    return fig


def heading(title, sub=""):
    st.markdown(f'<div class="h">{title}</div><div class="sub">{sub}</div>', unsafe_allow_html=True)


def spark(vals, color=ACCENT, w=120, h=26):
    vals = list(vals)
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1
    pts = " ".join(f"{i*w/(len(vals)-1):.1f},{h-2-(v-lo)/span*(h-4):.1f}" for i, v in enumerate(vals))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}"><polyline fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" points="{pts}"/></svg>')


def delta(vals, label="vs prior month"):
    if len(vals) < 2 or not vals[-2]:
        return ""
    d = vals[-1] / vals[-2] - 1
    return f'<span class="{"up" if d >= 0 else "down"}">{"▲" if d >= 0 else "▼"} {abs(d):.1%}</span> {label}'


def strip(items):
    cells = ""
    for it in items:
        l, v, sub = it[:3]
        sp = spark(it[3]) if len(it) > 3 else ""
        cells += f'<div><div class="l">{l}</div><div class="v">{v}</div><div class="s">{sub}</div>{sp}</div>'
    st.markdown(f'<div class="strip">{cells}</div>', unsafe_allow_html=True)


def gbp(x):
    return f"£{x/1e6:,.2f}M" if abs(x) >= 1e6 else f"£{x:,.0f}"


def plural(n, word):
    return f"{n:,} {word}" + ("" if n == 1 else "s")


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading Online Retail dataset...")
def load_data() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["InvoiceDate"])
    last_err = None
    for _ in range(4):  # the UCI download times out intermittently
        try:
            ds = fetch_ucirepo(id=352)
            break
        except Exception as e:
            last_err = e
            time.sleep(2)
    else:
        raise RuntimeError(f"Could not download dataset: {last_err}")
    df = pd.concat([ds.data.ids, ds.data.features], axis=1)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df.to_csv(CACHE, index=False)
    return df


@st.cache_data
def clean(df: pd.DataFrame):
    steps = [("Raw rows", len(df))]
    df = df.dropna(subset=["CustomerID"]).copy()
    steps.append(("With a customer ID", len(df)))
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    steps.append(("Excluding cancellations", len(df)))
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    steps.append(("Quantity and price > 0", len(df)))
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df, steps


@st.cache_data
def rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    return df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )


@st.cache_data
def scale(table: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(np.log1p(table[["Recency", "Frequency", "Monetary"]]))


@st.cache_data
def k_scan(scaled: np.ndarray, kmax=10):
    rows = []
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=10, random_state=42).fit(scaled)
        sil = silhouette_score(scaled, km.labels_, sample_size=min(2000, len(scaled)), random_state=42)
        rows.append((k, km.inertia_, sil))
    return pd.DataFrame(rows, columns=["k", "Inertia", "Silhouette"])


DESC = {
    "Champions": "Recent, frequent, highest spend",
    "Loyal": "Recent, regular buyers with solid value",
    "Promising": "Recent buyers, still few orders",
    "At risk": "Good past value, gone quiet",
    "Needs attention": "Inactive with below-average value",
    "Lost": "Long inactive, lowest value",
    "Regular": "Recent and steady, mid value",
    "New buyers": "Recent first or second orders",
    "Slipping": "Value fading, activity dropping",
    "Hibernating": "Inactive for a long time, low value",
}
ALT = {"Loyal": "Regular", "Promising": "New buyers", "At risk": "Slipping", "Needs attention": "Hibernating"}
PLAYBOOK = {
    "Champions": "Reward and ask for referrals",
    "Loyal": "Upsell and add loyalty perks",
    "Promising": "Nudge a second or third purchase",
    "At risk": "Win-back offer before they lapse",
    "Needs attention": "Time-limited offers to re-engage",
    "Lost": "Low-cost reactivation, otherwise sunset",
    "Regular": "Cross-sell related products",
    "New buyers": "Welcome series and a second-order incentive",
    "Slipping": "Personalised reminder with a small incentive",
    "Hibernating": "Low-cost reactivation email",
}


def name_clusters(table: pd.DataFrame, k: int):
    """Rank clusters by value, then name the middle ones from recency and value."""
    g = table.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    z = (g - g.mean()) / g.std(ddof=0).replace(0, 1)
    order = list((z["Frequency"] + z["Monetary"] - z["Recency"]).sort_values(ascending=False).index)
    med_rec, out, used = table["Recency"].median(), {}, set()
    for rank, c in enumerate(order):
        if k > 6:
            out[c] = (f"Segment {rank + 1}", "")
            continue
        if rank == 0:
            nm = "Champions"
        elif rank == k - 1:
            nm = "Lost"
        else:
            recent = g.Recency[c] <= med_rec
            high = (z.Frequency[c] + z.Monetary[c]) >= 0
            nm = {(True, True): "Loyal", (True, False): "Promising",
                  (False, True): "At risk", (False, False): "Needs attention"}[(recent, high)]
        if nm in used and nm in ALT:
            nm = ALT[nm]
        base_nm, n = nm, 2
        while nm in used:
            nm, n = f"{base_nm} {n}", n + 1
        used.add(nm)
        out[c] = (nm, DESC[base_nm])
    return out, order


raw = load_data()
df, steps = clean(raw)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="brand"><i></i>Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-h">Filters</div>', unsafe_allow_html=True)
    picked = st.multiselect("Country", sorted(df["Country"].unique()), placeholder="All countries")
    d0, d1 = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
    dates = st.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)

view = df
if picked:
    view = view[view["Country"].isin(picked)]
if len(dates) == 2:
    view = view[(view["InvoiceDate"].dt.date >= dates[0]) & (view["InvoiceDate"].dt.date <= dates[1])]
if view.empty:
    st.warning("No data for the current filters.")
    st.stop()

# ---------------------------------------------------------------- model (shared by all tabs)
table = rfm(view)
enough = len(table) >= 12
if enough:
    scaled = scale(table)
    scan = k_scan(scaled)
    best_k = int(scan.loc[scan["Silhouette"].idxmax(), "k"])
    with st.sidebar:
        st.markdown('<div class="side-h">Model</div>', unsafe_allow_html=True)
        k = st.slider("Number of segments (k)", 2, 8, min(max(best_k, 4), 6))
        st.caption(f"Statistically best k is {best_k} (silhouette); 4 or more gives more actionable segments.")
    km = KMeans(k, n_init=10, random_state=42).fit(scaled)
    table = table.assign(Cluster=km.labels_)
    info, order = name_clusters(table, k)
    table["Segment"] = table["Cluster"].map(lambda c: info[c][0])
    seg_order = [info[c][0] for c in order]
    seg_desc = {v[0]: v[1] for v in info.values()}
    color_map = {s: SEG_COLORS[i % len(SEG_COLORS)] for i, s in enumerate(seg_order)}
    sil = silhouette_score(scaled, km.labels_, sample_size=min(3000, len(scaled)), random_state=42)

# ---------------------------------------------------------------- header
st.markdown('<div class="page-title">Customer segmentation</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="page-sub">UCI Online Retail · {view["InvoiceDate"].min():%d %b %Y} to '
    f'{view["InvoiceDate"].max():%d %b %Y} · RFM features, K-Means clustering</div>',
    unsafe_allow_html=True,
)
if enough:
    st.markdown(
        f'<span class="pill">{len(table):,} customers</span><span class="pill">{k} segments</span>'
        f'<span class="pill">{len(view):,} transactions</span><br><br>',
        unsafe_allow_html=True,
    )

tab_over, tab_seg, tab_aud, tab_camp, tab_cust, tab_model = st.tabs(
    ["Overview", "Segments", "Audiences", "Campaigns", "Customers", "Model & data"]
)

# ---------------------------------------------------------------- overview
with tab_over:
    revenue, orders, custs = view["Revenue"].sum(), view["InvoiceNo"].nunique(), view["CustomerID"].nunique()
    repeat = (table["Frequency"] > 1).mean()
    mon = view.groupby(view["InvoiceDate"].dt.to_period("M")).agg(
        Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"), Customers=("CustomerID", "nunique"))
    mon["AOV"] = mon.Revenue / mon.Orders
    if view["InvoiceDate"].max().day < 28 and len(mon) > 1:  # drop the partial trailing month
        mon = mon.iloc[:-1]
    sv = {c: mon[c].tolist() for c in mon}
    strip([
        ("Revenue", gbp(revenue), delta(sv["Revenue"]) or "gross sales", sv["Revenue"]),
        ("Orders", f"{orders:,}", delta(sv["Orders"]) or "unique invoices", sv["Orders"]),
        ("Customers", f"{custs:,}", delta(sv["Customers"]) or "identified buyers", sv["Customers"]),
        ("Avg order value", f"£{revenue/orders:,.0f}", delta(sv["AOV"]) or "revenue per order", sv["AOV"]),
        ("Repeat customers", f"{repeat:.0%}", "placed 2+ orders"),
    ])

    if enough:
        pr = table.groupby("Segment").agg(n=("Recency", "size"), rev=("Monetary", "sum")).reindex(seg_order)
        top = pr.index[0]
        top_c, top_r = pr.n.iloc[0] / pr.n.sum(), pr.rev.iloc[0] / pr.rev.sum()
        tail = pr.iloc[-2:] if k >= 4 else pr.iloc[-1:]
        tail_rec = table[table.Segment.isin(tail.index)].Recency.mean()
        cs = view.groupby("Country")["Revenue"].sum()
        hr = view.groupby(view["InvoiceDate"].dt.hour)["Revenue"].sum().idxmax()
        dy = view.groupby(view["InvoiceDate"].dt.day_name())["Revenue"].sum().idxmax()
        finds = [
            ("Concentration", f"<b>{top}</b> are {top_c:.0%} of customers but drive <b>{top_r:.0%}</b> of revenue.", SEG_COLORS[0]),
            ("Win-back opportunity", f"<b>{int(tail.n.sum()):,}</b> customers in {' and '.join(tail.index)} last ordered ~<b>{tail_rec:.0f} days</b> ago and spent <b>{gbp(tail.rev.sum())}</b> to date.", SEG_COLORS[3]),
            ("Market and timing", f"<b>{cs.idxmax()}</b> brings {cs.max()/cs.sum():.0%} of revenue. Sales peak on <b>{dy}s</b> around <b>{hr}:00</b>.", SEG_COLORS[1]),
        ]
        st.markdown('<div class="finds">' + "".join(
            f'<div class="find" style="--c:{c}"><div class="k">{k_}</div><div class="t">{t}</div></div>' for k_, t, c in finds
        ) + "</div>", unsafe_allow_html=True)

    with st.container(border=True):
        heading("Monthly revenue", "The last month is partial in this dataset")
        m = view.set_index("InvoiceDate")["Revenue"].resample("MS").sum().reset_index()
        f = px.bar(m, x="InvoiceDate", y="Revenue", color_discrete_sequence=[ACCENT])
        cols = [ACCENT] * len(m)
        if view["InvoiceDate"].max().day < 28 and len(m) > 1:
            cols[-1] = "#B8C4F0"
        f.update_traces(marker_color=cols, hovertemplate="%{x|%b %Y}<br>£%{y:,.0f}<extra></extra>")
        f.update_yaxes(tickprefix="£", tickformat="~s", title=None)
        f.update_xaxes(title=None)
        st.plotly_chart(base(f, 300), width="stretch")

    l, r = st.columns(2)
    with l, st.container(border=True):
        heading("Top countries", "Revenue")
        tc = view.groupby("Country")["Revenue"].sum().nlargest(8).sort_values().reset_index()
        f = px.bar(tc, x="Revenue", y="Country", orientation="h", color_discrete_sequence=[ACCENT])
        f.update_xaxes(tickprefix="£", tickformat="~s", title=None)
        f.update_yaxes(title=None)
        st.plotly_chart(base(f, 330), width="stretch")
    with r, st.container(border=True):
        heading("Top products", "Revenue")
        tp = view.groupby("Description")["Revenue"].sum().nlargest(8).sort_values().reset_index()
        f = px.bar(tp, x="Revenue", y="Description", orientation="h", color_discrete_sequence=[ACCENT])
        f.update_xaxes(tickprefix="£", tickformat="~s", title=None)
        f.update_yaxes(title=None)
        st.plotly_chart(base(f, 330), width="stretch")

    with st.container(border=True):
        heading("Buying pattern", "Revenue by weekday and hour")
        t = view.assign(Day=view["InvoiceDate"].dt.day_name(), Hour=view["InvoiceDate"].dt.hour)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hm = t.pivot_table(index="Day", columns="Hour", values="Revenue", aggfunc="sum").reindex(days).dropna(how="all")
        f = px.imshow(hm, aspect="auto", color_continuous_scale=["#F1F4FD", ACCENT])
        f.update_coloraxes(showscale=False)
        f.update_xaxes(title="Hour of day", dtick=1)
        f.update_yaxes(title=None)
        st.plotly_chart(base(f, 280), width="stretch")

# ---------------------------------------------------------------- segments
with tab_seg:
    if not enough:
        st.warning("Not enough customers in the current filter for clustering.")
    else:
        strip([
            ("Segments", str(k), "K-Means clusters"),
            ("Customers clustered", f"{len(table):,}", "with purchase history"),
            ("Silhouette score", f"{sil:.2f}", "cluster separation, higher is better"),
            ("Top segment share", f"{table.groupby('Segment').Monetary.sum().max()/table.Monetary.sum():.0%}",
             "of revenue from one segment"),
        ])

        prof = table.groupby("Segment").agg(
            Customers=("Recency", "size"), Recency=("Recency", "mean"), Orders=("Frequency", "mean"),
            Spend=("Monetary", "mean"), Revenue=("Monetary", "sum"),
        ).reindex(seg_order)
        prof["Share of customers"] = prof.Customers / prof.Customers.sum() * 100
        prof["Share of revenue"] = prof.Revenue / prof.Revenue.sum() * 100

        cards = ""
        for name, rw in prof.iterrows():
            cards += (
                f'<div class="card" style="--c:{color_map[name]}"><div class="n">{name}</div>'
                f'<div class="big">{int(rw.Customers):,}<span>customers · {rw["Share of customers"]:.0f}%</span></div>'
                f'<div class="bar"><i style="width:{rw["Share of revenue"]:.0f}%"></i></div>'
                f'<div class="lbl">{rw["Share of revenue"]:.0f}% of revenue</div>'
                f'<div class="m"><span><b>{rw.Recency:.0f}d</b>recency</span><span><b>{rw.Orders:.1f}</b>orders</span>'
                f'<span><b>£{rw.Spend:,.0f}</b>avg spend</span></div>'
                f'<div class="act">→ {PLAYBOOK.get(name.rsplit(" ", 1)[0] if name[-1].isdigit() else name, seg_desc.get(name, ""))}</div></div>'
            )
        st.markdown(f'<div class="cards">{cards}</div>', unsafe_allow_html=True)

        with st.container(border=True):
            heading("Segment detail", "Sorted from most to least valuable customers")
            show = prof.reset_index()
            show.insert(1, "Profile", show["Segment"].map(seg_desc))
            st.dataframe(
                show[["Segment", "Profile", "Customers", "Share of customers", "Recency", "Orders", "Spend", "Share of revenue"]],
                hide_index=True, width="stretch",
                column_config={
                    "Customers": st.column_config.NumberColumn(format="%d"),
                    "Share of customers": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                    "Recency": st.column_config.NumberColumn("Avg recency (days)", format="%.0f"),
                    "Orders": st.column_config.NumberColumn("Avg orders", format="%.1f"),
                    "Spend": st.column_config.NumberColumn("Avg spend", format="£%.0f"),
                    "Share of revenue": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                },
            )

        l, r = st.columns(2)
        with l, st.container(border=True):
            heading("Customers vs revenue", "Where the money comes from compared with how many customers there are")
            long = prof[["Share of customers", "Share of revenue"]].reset_index().melt("Segment", var_name="Metric", value_name="Share")
            f = px.bar(long, y="Segment", x="Share", color="Metric", barmode="group", orientation="h",
                       color_discrete_map={"Share of customers": "#C5CCDA", "Share of revenue": ACCENT},
                       category_orders={"Segment": seg_order[::-1]})
            f.update_xaxes(ticksuffix="%")
            f.update_yaxes(title=None)
            st.plotly_chart(base(f, 360, True), width="stretch")
        with r, st.container(border=True):
            heading("Segment profile vs average", "100 = overall average; recency is better when lower")
            avg = table[["Recency", "Frequency", "Monetary"]].mean()
            idx = (prof[["Recency", "Orders", "Spend"]].set_axis(["Recency", "Frequency", "Monetary"], axis=1) / avg * 100).round(0)
            f = px.imshow(idx, text_auto=".0f", aspect="auto", color_continuous_scale=["#F1F4FD", ACCENT])
            f.update_coloraxes(showscale=False)
            f.update_xaxes(side="top", title=None)
            f.update_yaxes(title=None)
            st.plotly_chart(base(f, 360), width="stretch")

        l, r = st.columns(2)
        pca = PCA(2, random_state=42).fit(scaled)
        pts = pd.DataFrame(pca.transform(scaled), columns=["PC1", "PC2"], index=table.index).join(table["Segment"])
        with l, st.container(border=True):
            heading("Cluster map", f"PCA projection of the 3 RFM features, {pca.explained_variance_ratio_.sum():.0%} of variance")
            f = px.scatter(pts.reset_index(), x="PC1", y="PC2", color="Segment", color_discrete_map=color_map,
                           hover_name="CustomerID", opacity=0.7, category_orders={"Segment": seg_order})
            f.update_traces(marker=dict(size=5))
            st.plotly_chart(base(f, 360, True), width="stretch")
        with r, st.container(border=True):
            heading("Recency vs spend", "Each dot is a customer")
            f = px.scatter(table.reset_index(), x="Recency", y="Monetary", color="Segment", color_discrete_map=color_map,
                           hover_name="CustomerID", log_y=True, opacity=0.7, category_orders={"Segment": seg_order})
            f.update_traces(marker=dict(size=5))
            f.update_yaxes(title="Total spend (£, log)")
            f.update_xaxes(title="Days since last order")
            st.plotly_chart(base(f, 360, True), width="stretch")

        st.download_button("Download customer segments (CSV)", table.to_csv().encode(), "customer_segments.csv",
                           "text/csv", type="primary")

# ---------------------------------------------------------------- audiences
with tab_aud:
    if not enough:
        st.warning("Not enough customers in the current filter.")
    else:
        country_of = view.groupby("CustomerID")["Country"].agg(lambda x: x.mode().iat[0])
        cust = table.join(country_of)
        rmax, fmax, mmax = int(cust.Recency.max()), int(cust.Frequency.max()), float(cust.Monetary.max())
        PRESETS = {
            "Custom": dict(r=(0, rmax), f=(1, fmax), m=0.0),
            "High value (top 10% by spend)": dict(r=(0, rmax), f=(1, fmax), m=float(cust.Monetary.quantile(0.9))),
            "Win-back (inactive 90+ days, 2+ orders)": dict(r=(90, rmax), f=(2, fmax), m=0.0),
            "New and active (last 30 days, 1-2 orders)": dict(r=(0, 30), f=(1, 2), m=0.0),
        }
        with st.container(border=True):
            heading("Audience rules", "Customers must match every rule")
            preset = st.selectbox("Template", list(PRESETS), index=1)
            P = PRESETS[preset]
            c1, c2, c3, c4 = st.columns(4)
            rr = c1.slider("Days since last order", 0, rmax, P["r"], key=f"r_{preset}")
            ff = c2.slider("Number of orders", 1, fmax, P["f"], key=f"f_{preset}")
            mm = c3.slider("Total spend (£)", 0.0, mmax, (float(P["m"]), mmax), key=f"m_{preset}")
            cc = c4.multiselect("Country", sorted(cust.Country.unique()), placeholder="Any", key=f"c_{preset}")

        mask = cust.Recency.between(*rr) & cust.Frequency.between(*ff) & cust.Monetary.between(*mm)
        if cc:
            mask &= cust.Country.isin(cc)
        aud = cust[mask]
        strip([
            ("Audience size", f"{len(aud):,}", f"{len(aud)/len(cust):.1%} of customers"),
            ("Revenue in audience", gbp(aud.Monetary.sum()), f"{aud.Monetary.sum()/cust.Monetary.sum():.1%} of total"),
            ("Avg spend", f"£{aud.Monetary.mean():,.0f}" if len(aud) else "n/a", "per customer"),
            ("Avg days since order", f"{aud.Recency.mean():.0f}" if len(aud) else "n/a", "recency"),
        ])
        if len(aud):
            l, r = st.columns(2)
            with l, st.container(border=True):
                heading("Audience composition by segment", "How the rule set overlaps with the K-Means segments")
                comp = aud.Segment.value_counts().reindex(seg_order).dropna().reset_index()
                comp.columns = ["Segment", "Customers"]
                f = px.bar(comp, x="Customers", y="Segment", orientation="h", color="Segment",
                           color_discrete_map=color_map)
                f.update_yaxes(title=None)
                st.plotly_chart(base(f, 320), width="stretch")
            with r, st.container(border=True):
                heading("Audience by country", "Top 8")
                top = aud.Country.value_counts().head(8).iloc[::-1].reset_index()
                f = px.bar(top, x="count", y="Country", orientation="h", color_discrete_sequence=[ACCENT])
                f.update_xaxes(title="Customers")
                f.update_yaxes(title=None)
                st.plotly_chart(base(f, 320), width="stretch")
            with st.container(border=True):
                heading("Customers in audience", "Top 500 by spend")
                st.dataframe(aud.sort_values("Monetary", ascending=False).head(500).round(1),
                             width="stretch", height=300,
                             column_config={"Monetary": st.column_config.NumberColumn("Spend", format="£%.0f")})
            st.download_button("Export audience (CSV)", aud.to_csv().encode(), "audience.csv", "text/csv", type="primary")
        else:
            st.info("No customers match these rules.")

# ---------------------------------------------------------------- campaigns
with tab_camp:
    if not enough:
        st.warning("Not enough customers in the current filter.")
    else:
        l, r = st.columns([3, 2])
        with l, st.container(border=True):
            heading("Campaign setup", "Pick a segment, a channel and a message")
            target = st.selectbox("Target segment", seg_order)
            a, b = st.columns(2)
            channel = a.radio("Channel", ["Push", "Email", "SMS", "In-app"], horizontal=True)
            goal = b.radio("Objective", ["Win back", "Reward loyalty", "Upsell", "Welcome"], horizontal=True,
                           index=1 if target.startswith(("Champions", "Loyal")) else 0)
            TEMPL = {
                "Win back": ("We miss you", "It has been a while. Here is 15% off your next order, just for you."),
                "Reward loyalty": ("Thanks for being a top customer", "Get early access to new arrivals before everyone else."),
                "Upsell": ("Picked for you", "Customers like you love these. Add one more and get free delivery."),
                "Welcome": ("Welcome", "Start with 10% off your next purchase."),
            }
            title0, body0 = TEMPL[goal]
            title = st.text_input("Title", title0, key=f"t_{goal}")
            body = st.text_area("Message", body0, height=80, key=f"b_{goal}")
        with r, st.container(border=True):
            heading("Preview")
            st.markdown(
                f'<div class="phone"><div class="bar"></div><div class="notif"><div class="a">{channel} · now</div>'
                f'<div class="t">{title}</div><div class="b">{body}</div></div></div>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            heading("Assumptions", "Planning inputs you can adjust. These are not measured from the dataset")
            a, b, c3 = st.columns(3)
            deliver = a.slider("Delivery rate (%)", 50, 100, {"Push": 85, "Email": 95, "SMS": 98, "In-app": 60}[channel])
            ctr = b.slider("Click rate (%)", 1, 40, {"Push": 8, "Email": 3, "SMS": 6, "In-app": 12}[channel])
            conv = c3.slider("Click-to-order rate (%)", 1, 50, 15)

        seg_df = table[table.Segment == target]
        reach = len(seg_df) * deliver / 100
        clicks = reach * ctr / 100
        orders_est = clicks * conv / 100
        aov = view.groupby("InvoiceNo")["Revenue"].sum().mean()
        strip([
            ("Audience", f"{len(seg_df):,}", target),
            ("Delivered", f"{reach:,.0f}", f"{deliver}% of audience"),
            ("Clicks", f"{clicks:,.0f}", f"{ctr}% of delivered"),
            ("Orders", f"{orders_est:,.0f}", f"{conv}% of clicks"),
            ("Projected revenue", gbp(orders_est * aov), f"at £{aov:,.0f} avg order"),
        ])
        with st.container(border=True):
            heading("Projected funnel")
            fn = pd.DataFrame({"Stage": ["Audience", "Delivered", "Clicked", "Ordered"], "Customers": [len(seg_df), reach, clicks, orders_est]})
            f = px.bar(fn.iloc[::-1], x="Customers", y="Stage", orientation="h", text_auto=",.0f", color_discrete_sequence=[ACCENT])
            f.update_traces(textposition="outside", cliponaxis=False)
            f.update_yaxes(title=None)
            st.plotly_chart(base(f, 240), width="stretch")

# ---------------------------------------------------------------- customers
with tab_cust:
    ids = sorted(view.CustomerID.unique())
    multi = table.index[table.Frequency > 3].tolist()
    default = ids.index(multi[0]) if multi and multi[0] in ids else 0
    cid = st.selectbox("Customer ID", ids, index=default)
    h = view[view.CustomerID == cid]
    row = table.loc[cid]
    n_orders = h.InvoiceNo.nunique()
    strip([
        ("Segment", str(row["Segment"]) if enough else "n/a", "K-Means"),
        ("Total spend", f"£{h.Revenue.sum():,.0f}", plural(n_orders, "order")),
        ("Last order", f"{int(row.Recency)} days ago", f"{h.InvoiceDate.max():%d %b %Y}"),
        ("Country", h.Country.mode().iat[0], "most frequent"),
        ("Items bought", f"{h.Quantity.sum():,}", plural(h.StockCode.nunique(), "product")),
    ])
    l, r = st.columns(2)
    with l, st.container(border=True):
        heading("Orders", "Value per order")
        o = h.groupby("InvoiceNo").agg(Date=("InvoiceDate", "min"), Value=("Revenue", "sum")).reset_index()
        o["Order"] = o["Date"].dt.strftime("%d %b %Y")
        f = px.bar(o, x="Order", y="Value", color_discrete_sequence=[ACCENT])
        f.update_xaxes(type="category", title=None)
        f.update_yaxes(tickprefix="£")
        f.update_traces(width=0.4 if len(o) < 5 else None)
        st.plotly_chart(base(f, 320), width="stretch")
    with r, st.container(border=True):
        heading("Top products", "By spend")
        tp = h.groupby("Description")["Revenue"].sum().nlargest(8).sort_values().reset_index()
        f = px.bar(tp, x="Revenue", y="Description", orientation="h", color_discrete_sequence=[ACCENT])
        f.update_yaxes(title=None)
        f.update_xaxes(tickprefix="£")
        st.plotly_chart(base(f, 320), width="stretch")

# ---------------------------------------------------------------- model & data
with tab_model:
    l, r = st.columns(2)
    with l, st.container(border=True):
        heading("Data cleaning", "Rows kept after each step")
        s = pd.DataFrame(steps, columns=["Step", "Rows"])
        s["Kept"] = s["Rows"] / s["Rows"].iloc[0] * 100
        st.dataframe(s, hide_index=True, width="stretch", column_config={
            "Rows": st.column_config.NumberColumn(format="%d"),
            "Kept": st.column_config.ProgressColumn("Share of raw", format="%.0f%%", min_value=0, max_value=100),
        })
    with r, st.container(border=True):
        heading("Missing values in raw data", "Percent of rows")
        miss = (raw.isna().mean() * 100).round(2).rename("Missing").reset_index()
        miss.columns = ["Column", "Missing"]
        f = px.bar(miss[miss.Missing > 0], x="Missing", y="Column", orientation="h", color_discrete_sequence=[ACCENT])
        f.update_xaxes(ticksuffix="%")
        f.update_yaxes(title=None)
        st.plotly_chart(base(f, 190), width="stretch")

    if enough:
        with st.container(border=True):
            heading("Choosing k", "Elbow (inertia) and silhouette score for k = 2 to 10")
            a, b = st.columns(2)
            f = px.line(scan, x="k", y="Inertia", markers=True, color_discrete_sequence=[ACCENT])
            a.plotly_chart(base(f, 260), width="stretch")
            f = px.line(scan, x="k", y="Silhouette", markers=True, color_discrete_sequence=["#12A594"])
            f.add_vline(x=best_k, line_dash="dot", line_color=MUTED, annotation_text=f"best k = {best_k}")
            b.plotly_chart(base(f, 260), width="stretch")
        with st.container(border=True):
            heading("RFM feature distributions", "Log-scaled and standardised before clustering to reduce skew")
            c = st.columns(3)
            for col, feat in zip(c, ["Recency", "Frequency", "Monetary"]):
                f = px.histogram(table, x=feat, nbins=40, log_y=True, color_discrete_sequence=[ACCENT])
                col.markdown(f'<div class="sub">{feat}</div>', unsafe_allow_html=True)
                col.plotly_chart(base(f, 220), width="stretch")

    with st.container(border=True):
        heading("Transactions", "Search by product description")
        q = st.text_input("Search", placeholder="e.g. heart", label_visibility="collapsed")
        shown = view[view["Description"].astype(str).str.contains(q, case=False, na=False)] if q else view
        st.caption(f"{len(shown):,} rows, showing the first 2,000")
        st.dataframe(shown.head(2000), width="stretch", height=340)
