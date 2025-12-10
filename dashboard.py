import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="T2V Evaluation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR CLEAN UI ---
st.markdown("""
<style>
    [data-testid="column"] {
        border-right: 1px solid #e6e6e6;
        padding-right: 20px;
        padding-left: 20px;
    }
    [data-testid="column"]:last-child {
        border-right: none;
    }
    h3 {
        font-weight: 600 !important;
        font-size: 1.4rem !important;  /* Increased Header Size */
        margin-bottom: 20px !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important; /* Larger Metric Numbers */
    }
</style>
""", unsafe_allow_html=True)

# --- COLOR PALETTE ---
COLORS = {'Sora2': '#eb34bd', 'Veo3': '#349beb'}

# --- DATA LOADING ---
df_category = pd.DataFrame({
    'Category': ['Biological', 'Physical', 'Social', 'Temporal'],
    'Sora2': [74.8, 88.9, 97.0, 74.3],
    'Veo3': [73.0, 86.1, 95.1, 65.0]
}).melt(id_vars='Category', var_name='Model', value_name='Score')

df_radar = pd.DataFrame({
    'Type': ['Action', 'Audio', 'Environment', 'Subject'],
    'Sora2': [71.8, 81.7, 95.0, 93.5],
    'Veo3': [64.5, 77.6, 94.5, 92.5]
})

# --- TITLE SECTION ---
st.title("🎬 T2V Model Evaluation: Unconventional Prompts")
st.markdown("##### Forensic analysis of **Sora2** vs **Veo3** across adversarial categories.")
st.divider()

# --- ROW 1: HEADLINE METRICS & CATEGORY BREAKDOWN ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🏆 Global Alignment")

    m1, m2 = st.columns(2)
    m1.metric("Sora 2", "84.5%", delta="Winner", delta_color="normal")
    m2.metric("Veo 3", "81.0%", delta="-3.5%", delta_color="normal")

    df_global = pd.DataFrame({'Model': ['Sora2', 'Veo3'], 'Score': [84.5, 81.0]})
    fig_global = px.bar(
        df_global, x='Model', y='Score', color='Model',
        color_discrete_map=COLORS, text_auto='.1f'
    )
    fig_global.update_layout(
        showlegend=False,
        yaxis_range=[0, 100],
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        font=dict(size=14)  # Base font increase
    )
    # Increased text size on bars
    fig_global.update_traces(textfont_size=20, textangle=0, textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_global, use_container_width=True)

with col2:
    st.subheader("📊 Category Weakness Breakdown")
    fig_cat = px.bar(
        df_category, x='Category', y='Score', color='Model',
        barmode='group', color_discrete_map=COLORS, text_auto='.1f'
    )
    fig_cat.update_layout(
        legend_title=None,
        yaxis_range=[0, 115],
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=14)),
        height=350,
        font=dict(size=14)  # Base font increase
    )
    # Increased text size on bars
    fig_cat.update_traces(textfont_size=14, textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_cat, use_container_width=True)

st.divider()

# --- ROW 2: DEEP DIVE METRICS ---
c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.subheader("🕸️ Capability Radar")

    fig_radar = go.Figure()
    categories = df_radar['Type'].tolist()

    fig_radar.add_trace(go.Scatterpolar(
        r=df_radar['Sora2'], theta=categories,
        fill='toself', name='Sora2',
        line_color=COLORS['Sora2'], opacity=0.7
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=df_radar['Veo3'], theta=categories,
        fill='toself', name='Veo3',
        line_color=COLORS['Veo3'], opacity=0.6
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[50, 100], tickfont=dict(color='#666666', size=10)),
            angularaxis=dict(tickfont=dict(color='black', size=14, weight="bold"))
        ),
        showlegend=True,
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", y=-0.15, font=dict(size=14)),
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with c2:
    st.subheader("📉 Prior Bias (Object vs Action)")
    st.caption("Drop-off from generating the *Subject* to making it perform the *Action*.")

    fig_slope = go.Figure()

    models = ['Sora2', 'Veo3']
    subj = [93.5, 92.5]
    act = [71.8, 64.5]

    for i, m in enumerate(models):
        c = COLORS[m]

        if i == 0:  # Sora2
            text_pos = ["top center", "top center"]
            annotate = "Sora2 Gap:"
        else:  # Veo3
            text_pos = ["bottom center", "bottom center"]
            annotate = "Veo3.1 Gap:"
        fig_slope.add_trace(go.Scatter(
            x=['Subject', 'Action'], y=[subj[i], act[i]],
            mode='lines+markers+text',
            name=m,
            line=dict(color=c, width=5, dash='dot'),  # Thicker line
            marker=dict(size=14),  # Larger dots
            text=[f"{subj[i]}%", f"{act[i]}%"],
            textfont=dict(size=14, weight="bold"),  # Larger label text
            textposition=text_pos
        ))

        mid_y = (subj[i] + act[i]) / 2
        y_offset = 4 if i == 0 else -4

        fig_slope.add_annotation(
            x=0.5, y=mid_y + y_offset,
            text=f"{annotate}\n-{subj[i] - act[i]:.1f}%",
            showarrow=False,
            font=dict(color='#666666', size=14, weight="bold"),  # Larger Annotation
            bgcolor="white"
        )

    fig_slope.update_layout(
        yaxis=dict(range=[50, 100], showgrid=True, tickfont=dict(size=12)),
        xaxis=dict(showgrid=False, tickfont=dict(size=14)),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=40),
        height=400
    )
    st.plotly_chart(fig_slope, use_container_width=True)

with c3:
    st.subheader("🔊 Action vs Audio Gap")
    st.caption("Comparing physical action accuracy vs sound generation.")

    fig_av_slope = go.Figure()

    models = ['Sora2', 'Veo3']
    action_scores = [71.8, 64.5]
    audio_scores = [81.7, 77.6]
    i=0
    for i, m in enumerate(models):
        c = COLORS[m]

        if i == 0:
            text_pos = ["top center", "top center"]
            annotate = "Sora2 Gap:"
        else:
            text_pos = ["bottom center", "bottom center"]
            annotate = "Veo3.1 Gap:"
        fig_av_slope.add_trace(go.Scatter(
            x=['Action', 'Audio'], y=[action_scores[i], audio_scores[i]],
            mode='lines+markers+text',
            name=m,
            line=dict(color=c, width=5, dash='dot'),
            marker=dict(size=14),
            text=[f"{action_scores[i]}%", f"{audio_scores[i]}%"],
            textfont=dict(size=14, weight="bold"),
            textposition=text_pos
        ))

        mid_y = (action_scores[i] + audio_scores[i]) / 2
        y_offset = 4 if i == 0 else -4

        diff = audio_scores[i] - action_scores[i]
        diff_text = f"{annotate}\n+{diff:.1f}%"

        fig_av_slope.add_annotation(
            x=0.5, y=mid_y + y_offset,
            text=diff_text,
            showarrow=False,
            font=dict(color='#666666', size=14, weight="bold"),
            bgcolor="white"
        )

    fig_av_slope.update_layout(
        yaxis=dict(range=[50, 100], showgrid=True, tickfont=dict(size=12)),
        xaxis=dict(showgrid=False, tickfont=dict(size=14)),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=40),
        height=400
    )
    st.plotly_chart(fig_av_slope, use_container_width=True)