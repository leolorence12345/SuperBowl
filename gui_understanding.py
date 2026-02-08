"""
Streamlit GUI for match event analysis (knockdowns, fouls, goals, etc.)
in a selected time window. Uses understanding.py + Gemini.
"""
import streamlit as st
from understanding import analyze_video, analyze_video_stream

st.set_page_config(
    page_title="Match Event Analysis",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ Match Event Analysis")
st.caption("Find significant events (goals, fouls, knockdowns, cards, etc.) in a time window")

with st.form("analysis_form"):
    video_uri = st.text_input(
        "Video URL",
        value="https://www.youtube.com/watch?v=CgCJ2nBAEaU",
        help="YouTube or other video URL supported by Gemini",
    )
    col1, col2 = st.columns(2)
    with col1:
        start_time = st.text_input("Start time (e.g. 10:00 or 600 seconds)", value="10:00")
    with col2:
        end_time = st.text_input("End time (e.g. 10:20 or 620 seconds)", value="10:20")
    stream_output = st.checkbox("Stream response (show text as it’s generated)", value=True)
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not video_uri or not start_time or not end_time:
        st.error("Please fill in video URL, start time, and end time.")
    else:
        with st.spinner("Analyzing video with Gemini…"):
            try:
                if stream_output:
                    placeholder = st.empty()
                    full_text = []
                    for chunk in analyze_video_stream(video_uri, start_time, end_time):
                        full_text.append(chunk)
                        placeholder.markdown("".join(full_text))
                else:
                    text = analyze_video(video_uri, start_time, end_time)
                    st.markdown(text)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
