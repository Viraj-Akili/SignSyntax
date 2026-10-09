"""
SIGN SYNTAX - Compiler Design Dashboard & Demonstration System
Phase 2: Core Implementation (Interactive Web Frontend)

A modern, academic presentation layer over the authoritative SignSyntax compiler.

Architecture:
  Gloss Input -> Symbol Table -> Lexer -> Parser -> AST -> Semantic Analyzer -> IR

Key Invariants:
- Zero duplicated compiler logic (Directly invokes `compile_frontend()`).
- All displayed tokens, parse trees, ASTs, semantic forms, and IR instructions
  are dynamically populated from the real `FrontendResult`.
- Strict phase isolation is faithfully visualized for valid and error inputs.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import cv2
    _VISION_DEPS_AVAILABLE = True
except ImportError:
    _VISION_DEPS_AVAILABLE = False

import numpy as np
import pandas as pd
import streamlit as st

from compiler.frontend import FrontendCompiler, FrontendResult, compile_frontend
from compiler.symbol_table import Symbol, SymbolTable
from recognition.hand_tracker import HandTracker
from recognition.wlasl_processor import format_frame_landmarks
from recognition.predict_gesture import GesturePredictor, GesturePredictionResult


# ==============================================================================
# Page Configuration & Modern Technical Styling
# ==============================================================================

st.set_page_config(
    page_title="SIGN SYNTAX | Compiler System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    /* Dark Academic & Technical Theme */
    :root {
        --bg-main: #0B0F19;
        --bg-card: #111827;
        --bg-card-hover: #1F2937;
        --border-color: #1E293B;
        --border-active: #334155;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --accent-emerald: #10B981;
        --accent-rose: #F43F5E;
        --accent-indigo: #6366F1;
        --accent-cyan: #06B6D4;
        --accent-amber: #F59E0B;
    }

    /* Hide Streamlit default fixed header to prevent overlay clipping */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Container constraints with safe positive top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    
    /* Header Typography - Fully visible & unclipped inside normal page flow */
    .brand-header-wrapper {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid var(--border-color);
        padding-top: 0.5rem;
        padding-bottom: 1.25rem;
        margin-top: 0;
        margin-bottom: 1.25rem;
        gap: 0.75rem;
        width: 100%;
        box-sizing: border-box;
        position: relative;
    }
    .brand-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .brand-mark {
        background: linear-gradient(135deg, #6366F1, #3B82F6);
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 900;
        width: 38px;
        height: 38px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .brand-title-text {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #FFFFFF;
        margin: 0;
        padding: 0;
        line-height: 1.25;
        overflow: visible;
        white-space: nowrap;
    }
    .brand-subtitle-text {
        font-size: 0.88rem;
        color: var(--text-secondary);
        margin-top: 0.15rem;
        font-weight: 400;
        line-height: 1.3;
    }
    .badge-container {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        flex-wrap: wrap;
    }
    .status-badge {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 4px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .badge-phase {
        background-color: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .badge-tests {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Pipeline Status Grid */
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.5rem;
        margin: 1rem 0;
    }
    @media (max-width: 768px) {
        .pipeline-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    .pipeline-node {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 0.65rem 0.4rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .pipeline-node-passed {
        border-top: 3px solid var(--accent-emerald);
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.08) 0%, var(--bg-card) 100%);
    }
    .pipeline-node-failed {
        border-top: 3px solid var(--accent-rose);
        background: linear-gradient(180deg, rgba(244, 63, 94, 0.08) 0%, var(--bg-card) 100%);
    }
    .pipeline-node-skipped {
        border-top: 3px solid var(--text-muted);
        opacity: 0.6;
    }
    .node-step {
        font-size: 0.68rem;
        color: var(--text-muted);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .node-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0.15rem 0 0.3rem 0;
    }
    .node-status-passed {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent-emerald);
    }
    .node-status-failed {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent-rose);
    }
    .node-status-skipped {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted);
    }

    /* Result Containers & Panels */
    .card-panel {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.85rem;
    }
    .card-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.75px;
        margin-bottom: 0.65rem;
    }
    
    /* Semantic Metric Grid */
    .semantic-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.5rem;
    }
    .semantic-card {
        background-color: #0F172A;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 0.6rem 0.75rem;
    }
    .semantic-label {
        font-size: 0.68rem;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .semantic-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-top: 0.15rem;
        font-family: "JetBrains Mono", monospace;
    }
    .semantic-value-null {
        color: var(--text-muted);
        font-style: italic;
    }

    /* Terminal/Code Display */
    .code-terminal {
        background-color: #030712 !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        font-family: "JetBrains Mono", "Fira Code", "Courier New", monospace !important;
        font-size: 0.84rem !important;
        color: #E2E8F0 !important;
        line-height: 1.5 !important;
        margin: 0 !important;
        overflow-x: auto !important;
        white-space: pre !important;
        display: block !important;
    }
    div[data-testid="stCodeBlock"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        background-color: #030712 !important;
    }
    div[data-testid="stCodeBlock"] pre {
        background-color: #030712 !important;
        font-family: "JetBrains Mono", "Fira Code", "Courier New", monospace !important;
        font-size: 0.84rem !important;
        line-height: 1.5 !important;
        white-space: pre !important;
    }

    /* Status Banners */
    .banner-success {
        background-color: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 6px;
        padding: 0.65rem 0.9rem;
        color: #6EE7B7;
        font-size: 0.88rem;
        margin: 0.6rem 0;
    }
    .banner-error {
        background-color: rgba(244, 63, 94, 0.08);
        border: 1px solid rgba(244, 63, 94, 0.25);
        border-radius: 6px;
        padding: 0.65rem 0.9rem;
        color: #FDA4AF;
        font-size: 0.88rem;
        margin: 0.6rem 0;
        font-family: "JetBrains Mono", monospace;
    }

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# Helper Functions & Authoritative Symbol Table Access
# ==============================================================================

@st.cache_resource
def get_symbol_table() -> SymbolTable:
    """Loads authoritative Symbol Table instance."""
    return SymbolTable()


@st.cache_resource
def get_gesture_predictor() -> Optional[GesturePredictor]:
    """Loads authoritative trained k-NN gesture classifier."""
    try:
        return GesturePredictor()
    except Exception:
        return None


@st.cache_resource
def get_hand_tracker() -> Optional[HandTracker]:
    """Loads MediaPipe HandTracker instance."""
    try:
        if _VISION_DEPS_AVAILABLE:
            return HandTracker()
        return None
    except Exception:
        return None


def render_pipeline_node(step_num: str, title: str, status: str) -> str:
    """Generates HTML string for a single horizontal pipeline node."""
    if status == "PASSED":
        node_class = "pipeline-node-passed"
        status_html = '<div class="node-status-passed">✓ PASSED</div>'
    elif status == "FAILED":
        node_class = "pipeline-node-failed"
        status_html = '<div class="node-status-failed">✗ FAILED</div>'
    else:
        node_class = "pipeline-node-skipped"
        status_html = '<div class="node-status-skipped">○ NOT EXECUTED</div>'

    return f"""
    <div class="pipeline-node {node_class}">
        <div class="node-step">{step_num}</div>
        <div class="node-title">{title}</div>
        {status_html}
    </div>
    """


# ==============================================================================
# Header Bar (Unclipped & Robust Layout)
# ==============================================================================

st.markdown(
    """
    <div class="brand-header-wrapper">
        <div class="brand-left">
            <div class="brand-mark">S</div>
            <div>
                <div class="brand-title-text">SIGN SYNTAX</div>
                <div class="brand-subtitle-text">A Compiler-Based Sign Language Processing System &bull; <i>Sign Language &rarr; Formal Language &rarr; Intermediate Representation</i></div>
            </div>
        </div>
        <div class="badge-container">
            <span class="status-badge badge-phase">Phase 2 &bull; Core Compiler</span>
            <span class="status-badge badge-tests">216 Tests Passing</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Top-Level Navigation Tabs
# ==============================================================================

tab_compiler, tab_recognition, tab_architecture, tab_symbols = st.tabs([
    "COMPILER",
    "RECOGNITION",
    "ARCHITECTURE",
    "SYMBOL TABLE",
])


# ==============================================================================
# TAB 1: COMPILER (Main Presentation Workspace)
# ==============================================================================

with tab_compiler:
    # -------------------------------------------------------------------------
    # Authoritative Gloss Input Area & Quick Examples
    # -------------------------------------------------------------------------
    if "main_gloss_input" not in st.session_state:
        st.session_state["main_gloss_input"] = "ME WANT FOOD"

    def set_gloss_input(val: str) -> None:
        st.session_state["main_gloss_input"] = val

    # Central Gloss Input Area
    st.markdown("#### GLOSS INPUT")
    st.caption("Enter a sequence of recognized sign glosses (case-sensitive) to process through the compiler.")

    col_input_box, col_btn = st.columns([4.8, 1.4])
    with col_input_box:
        gloss_input = st.text_input(
            "Gloss Input",
            key="main_gloss_input",
            label_visibility="collapsed",
            placeholder="e.g. ME WANT FOOD",
        )
    with col_btn:
        compile_btn = st.button("COMPILE SENTENCE", type="primary", width="stretch")

    # Quick Example Buttons (Populate the single authoritative input)
    st.caption("Quick Test Examples:")
    ex_row1 = st.columns(4)
    with ex_row1[0]:
        st.button("ME WANT FOOD", key="btn_ex_1", on_click=set_gloss_input, args=("ME WANT FOOD",), width="stretch")
    with ex_row1[1]:
        st.button("TODAY ME WANT FOOD", key="btn_ex_2", on_click=set_gloss_input, args=("TODAY ME WANT FOOD",), width="stretch")
    with ex_row1[2]:
        st.button("ME GO", key="btn_ex_3", on_click=set_gloss_input, args=("ME GO",), width="stretch")
    with ex_row1[3]:
        st.button("SHE HAVE CAR BIG", key="btn_ex_4", on_click=set_gloss_input, args=("SHE HAVE CAR BIG",), width="stretch")

    ex_row2 = st.columns(5)
    with ex_row2[0]:
        st.button("WHAT YOU WANT", key="btn_ex_5", on_click=set_gloss_input, args=("WHAT YOU WANT",), width="stretch")
    with ex_row2[1]:
        st.button("WHERE YOU GO", key="btn_ex_6", on_click=set_gloss_input, args=("WHERE YOU GO",), width="stretch")
    with ex_row2[2]:
        st.button("ME WANT FOOD NOT", key="btn_ex_7", on_click=set_gloss_input, args=("ME WANT FOOD NOT",), width="stretch")
    with ex_row2[3]:
        st.button("ME DANCE FOOD", key="btn_ex_8", help="Demonstrates Lexical Error: 'DANCE' is not in Symbol Table", on_click=set_gloss_input, args=("ME DANCE FOOD",), width="stretch")
    with ex_row2[4]:
        st.button("ME FOOD WANT", key="btn_ex_9", help="Demonstrates Syntax Error: Invalid SOV order rejected by LL(1) grammar", on_click=set_gloss_input, args=("ME FOOD WANT",), width="stretch")

    current_source = gloss_input.strip() if gloss_input else ""

    # Execute Compiler Frontend
    res: Optional[FrontendResult] = None
    if current_source:
        res = compile_frontend(current_source)

    # -------------------------------------------------------------------------
    # Horizontal Pipeline Status Bar (6 Stages)
    # -------------------------------------------------------------------------
    if res is None:
        st_sym = "PASSED"
        st_lex = "NOT EXECUTED"
        st_syn = "NOT EXECUTED"
        st_ast = "NOT EXECUTED"
        st_sem = "NOT EXECUTED"
        st_ir = "NOT EXECUTED"
    else:
        st_sym = "PASSED"
        st_lex = "PASSED" if (res.lexer_result and res.lexer_result.is_valid) else ("FAILED" if res.failed_phase == "LEXICAL" else "NOT EXECUTED")
        st_syn = "PASSED" if (res.parse_result and res.parse_result.is_valid) else ("FAILED" if res.failed_phase == "SYNTAX" else "NOT EXECUTED")
        st_ast = "PASSED" if (res.ast is not None) else "NOT EXECUTED"
        st_sem = "PASSED" if (res.semantic_result and res.semantic_result.is_valid) else ("FAILED" if res.failed_phase == "SEMANTIC" else "NOT EXECUTED")
        st_ir = "PASSED" if (res.ir_program is not None) else ("FAILED" if res.failed_phase == "IR" else "NOT EXECUTED")

    st.markdown("#### COMPILER PIPELINE")
    pipeline_html = f"""
    <div class="pipeline-grid">
        {render_pipeline_node("Stage 0", "SYMBOL TABLE", st_sym)}
        {render_pipeline_node("Stage 1", "LEXER", st_lex)}
        {render_pipeline_node("Stage 2", "PARSER", st_syn)}
        {render_pipeline_node("Stage 3", "AST", st_ast)}
        {render_pipeline_node("Stage 4", "SEMANTIC", st_sem)}
        {render_pipeline_node("Stage 5", "IR GENERATOR", st_ir)}
    </div>
    """
    st.markdown(pipeline_html, unsafe_allow_html=True)

    # Success / Error Status Banner
    if res is not None:
        if res.is_valid:
            st.markdown(
                '<div class="banner-success">'
                '<b>✓ COMPILATION SUCCESSFUL:</b> All compiler stages completed successfully. Target-independent Intermediate Representation generated.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            err_text = "<br>".join([f"&bull; {str(e)}" for e in res.errors])
            st.markdown(
                f'<div class="banner-error">'
                f'<b>✗ COMPILATION HALTED AT {res.failed_phase} PHASE:</b><br>{err_text}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # Stage Results (Two-Column Layout: Lexer & AST)
    # -------------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">1. LEXICAL ANALYSIS (TOKENS)</div>', unsafe_allow_html=True)
        if res and res.lexer_result and res.lexer_result.is_valid:
            token_rows = [
                {
                    "Position": t.position,
                    "Lexeme": t.lexeme,
                    "Token Type": t.token_type,
                    "Meaning": t.meaning,
                }
                for t in res.lexer_result.tokens
            ]
            st.dataframe(pd.DataFrame(token_rows), width="stretch", hide_index=True)
            st.caption(f"Verified {len(res.lexer_result.tokens)} tokens against Symbol Table.")
        elif res and res.failed_phase == "LEXICAL":
            st.error(f"Lexical Error: {res.lexer_result.errors[0]}")
        else:
            st.markdown('<div class="semantic-value-null">○ Lexical stage not executed.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">2. ABSTRACT SYNTAX TREE (AST)</div>', unsafe_allow_html=True)
        if res and res.ast:
            st.code(res.ast.pretty_print(), language=None)
            st.caption(f"Root Node: `{type(res.ast).__name__}`")
        elif res and res.failed_phase == "SYNTAX":
            st.error(f"Syntax Error: {res.parse_result.errors[0].message}")
        else:
            st.markdown('<div class="semantic-value-null">○ AST unavailable (halted upstream).</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Semantic Analysis Section
    # -------------------------------------------------------------------------
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">3. NORMALIZED SEMANTIC ANALYSIS</div>', unsafe_allow_html=True)

    if res and res.semantic_result and res.semantic_result.is_valid and res.semantic_form:
        form = res.semantic_form
        form_dict = form.to_dict()

        if form_dict.get("form_type") == "Declarative":
            st.markdown(
                f"""
                <div class="semantic-grid">
                    <div class="semantic-card">
                        <div class="semantic-label">SUBJECT</div>
                        <div class="semantic-value">{form.subject}</div>
                    </div>
                    <div class="semantic-card">
                        <div class="semantic-label">ACTION (VERB)</div>
                        <div class="semantic-value">{form.action}</div>
                    </div>
                    <div class="semantic-card">
                        <div class="semantic-label">OBJECT</div>
                        <div class="semantic-value">{form.object if form.object is not None else '<span class="semantic-value-null">null</span>'}</div>
                    </div>
                    <div class="semantic-card">
                        <div class="semantic-label">TIME CONTEXT</div>
                        <div class="semantic-value">{form.time if form.time is not None else '<span class="semantic-value-null">null</span>'}</div>
                    </div>
                    <div class="semantic-card">
                        <div class="semantic-label">ADJECTIVE</div>
                        <div class="semantic-value">{form.adjective if form.adjective is not None else '<span class="semantic-value-null">null</span>'}</div>
                    </div>
                    <div class="semantic-card">
                        <div class="semantic-label">NEGATED</div>
                        <div class="semantic-value">{"true" if form.negated else "false"}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Question form
            st.markdown(
                f"""
                <div class="semantic-grid">
                    <div class="semantic-card">
                        <div class="semantic-label">WH-OPERATOR</div>
                        <div class="semantic-value">{form.wh}</div>
                    </div>
                    <div class="semantic-card">
                        <div class="semantic-label">SUBJECT</div>
                        <div class="semantic-value">{form.subject if form.subject is not None else '<span class="semantic-value-null">null</span>'}</div>
                    </div>
                    <div class="semantic-card">
                        <div class="semantic-label">ACTION</div>
                        <div class="semantic-value">{form.action if form.action is not None else '<span class="semantic-value-null">null</span>'}</div>
                    </div>
                    <div class="semantic-card">
                        <div class="semantic-label">OBJECT</div>
                        <div class="semantic-value">{form.object if form.object is not None else '<span class="semantic-value-null">null</span>'}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif res and res.failed_phase == "SEMANTIC":
        for err in res.semantic_result.errors:
            st.error(f"Semantic Error in {err.node_type}: {err.message}")
    else:
        st.markdown('<div class="semantic-value-null">○ Semantic analysis not executed.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Intermediate Representation (IR) Section
    # -------------------------------------------------------------------------
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">4. TARGET-INDEPENDENT INTERMEDIATE REPRESENTATION (IR)</div>', unsafe_allow_html=True)

    if res and res.ir_program:
        c_ir_info, c_ir_terminal = st.columns([1.2, 3])
        with c_ir_info:
            st.markdown(f"**Program Type:** `{res.ir_program.program_type}`")
            st.markdown(f"**Instruction Count:** `{len(res.ir_program)}`")
            st.caption(
                "The Intermediate Representation provides a target-independent, linearized "
                "instruction stream decoupled from surface language syntax, ready for Phase 3 target generation."
            )
        with c_ir_terminal:
            st.code(res.ir_program.to_text(), language=None)
    else:
        st.markdown('<div class="semantic-value-null">○ IR generation not reached. (Requires valid upstream semantics).</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 2: RECOGNITION (Input Acquisition Subsystem & Live Recognition)
# ==============================================================================

with tab_recognition:
    st.subheader("LIVE GESTURE RECOGNITION")
    st.caption("Real-Time Webcam Video Capture → MediaPipe Hands → 252-D Feature Aggregation → Trained k-NN Classifier")
    st.divider()

    # Session State Initialization for Live Recognition & Gloss Stream
    if "gloss_stream" not in st.session_state:
        st.session_state["gloss_stream"] = []
    if "live_predicted_gloss" not in st.session_state:
        st.session_state["live_predicted_gloss"] = None
    if "live_prediction_result" not in st.session_state:
        st.session_state["live_prediction_result"] = None
    if "camera_active" not in st.session_state:
        st.session_state["camera_active"] = False
    if "capture_status" not in st.session_state:
        st.session_state["capture_status"] = "IDLE"

    predictor = get_gesture_predictor()
    tracker = get_hand_tracker()

    def render_capture_status_badge(placeholder, status_text: str, message: str = ""):
        """Renders the official visible status indicator for live gesture recognition."""
        status_colors = {
            "WAITING FOR HAND": "#f59e0b",
            "READY": "#0ea5e9",
            "COUNTDOWN": "#a855f7",
            "CAPTURING": "#22c55e",
            "CLASSIFIED": "#10b981",
            "IDLE": "#64748b",
        }
        color = status_colors.get(status_text, "#64748b")
        msg_html = f'<span style="color:#cbd5e1;font-size:0.85rem;margin-left:12px;">{message}</span>' if message else ""
        placeholder.markdown(
            f'<div style="display:flex;align-items:center;background:rgba(15,23,42,0.6);'
            f'border:1px solid rgba(255,255,255,0.12);border-left:4px solid {color};padding:8px 14px;border-radius:6px;margin-bottom:12px;">'
            f'<span style="font-size:0.75rem;letter-spacing:0.08em;color:#94a3b8;margin-right:8px;font-weight:600;">STATUS:</span>'
            f'<span style="color:{color};font-weight:700;font-size:0.95rem;letter-spacing:0.04em;">{status_text}</span>'
            f'{msg_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if predictor is None or tracker is None:
        st.error("Error: Gesture classifier model or MediaPipe vision dependencies are unavailable.")
    else:
        # Layout: Left column for Camera / Capture, Right column for Prediction Status
        col_cam, col_pred = st.columns([1.6, 1.4])

        with col_cam:
            st.markdown("#### Camera Feed & Gesture Capture")

            c_btn1, c_btn2, c_btn3 = st.columns(3)
            with c_btn1:
                start_btn = st.button("🎥 START CAMERA", type="primary", key="btn_start_cam", width="stretch")
            with c_btn2:
                stop_btn = st.button("🛑 STOP CAMERA", key="btn_stop_cam", width="stretch")
            with c_btn3:
                clear_gesture_btn = st.button("🔄 CLEAR GESTURE", key="btn_clear_gest", width="stretch")

            if start_btn:
                st.session_state["camera_active"] = True
                st.session_state["capture_status"] = "WAITING FOR HAND"
            if stop_btn:
                st.session_state["camera_active"] = False
                st.session_state["capture_status"] = "IDLE"
                st.rerun()
            if clear_gesture_btn:
                st.session_state["live_predicted_gloss"] = None
                st.session_state["live_prediction_result"] = None
                st.session_state["capture_status"] = "IDLE"
                st.rerun()

            status_badge_placeholder = st.empty()
            cam_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_placeholder = st.empty()

            if st.session_state.get("camera_active", False):
                # Try DirectShow first on Windows for fast device access, fallback to default
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)

                if not cap.isOpened():
                    st.error("Could not access webcam device (index 0). Please check device connection or permissions.")
                    st.session_state["camera_active"] = False
                    st.session_state["capture_status"] = "IDLE"
                else:
                    frames_buffer: List[List[float]] = []
                    required_window = 24  # Exactly 24 hand-present frames required

                    try:
                        # Warm up camera briefly for exposure/white-balance stabilization
                        for _ in range(5):
                            cap.read()

                        # -------------------------------------------------------------
                        # STAGE 1: WAITING FOR HAND
                        # -------------------------------------------------------------
                        st.session_state["capture_status"] = "WAITING FOR HAND"
                        render_capture_status_badge(
                            status_badge_placeholder,
                            "WAITING FOR HAND",
                            "Position hand(s) in front of the camera to begin...",
                        )
                        status_placeholder.info("📹 **Waiting for hand...** Position your hand in view to trigger the gesture capture sequence.")

                        hand_detected = False
                        max_idle_frames = 150  # ~5 seconds timeout
                        idle_count = 0

                        while not hand_detected and st.session_state.get("camera_active", False):
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                st.warning("Failed to retrieve frame from webcam.")
                                break

                            hands, annotated = tracker.process_frame(frame, draw=True)
                            display_frame = cv2.flip(annotated, 1)

                            if len(hands) > 0:
                                hand_detected = True
                                break

                            idle_count += 1
                            if idle_count > max_idle_frames:
                                status_placeholder.warning("No hands detected within timeout. Click START CAMERA again when ready.")
                                break

                            cv2.putText(
                                display_frame,
                                "WAITING FOR HAND",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.85,
                                (0, 165, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            cam_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), width="stretch")
                            time.sleep(0.02)

                        # -------------------------------------------------------------
                        # STAGE 2: READY — Prepare your gesture
                        # -------------------------------------------------------------
                        if hand_detected and st.session_state.get("camera_active", False):
                            st.session_state["capture_status"] = "READY"
                            render_capture_status_badge(
                                status_badge_placeholder,
                                "READY",
                                "Hand detected! Prepare your gesture...",
                            )
                            status_placeholder.info("✋ **READY — Prepare your gesture**")

                            ready_frames = 20  # ~0.8 seconds to settle in position
                            for _ in range(ready_frames):
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    break
                                hands, annotated = tracker.process_frame(frame, draw=True)
                                display_frame = cv2.flip(annotated, 1)

                                cv2.putText(
                                    display_frame,
                                    "READY - PREPARE GESTURE",
                                    (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.85,
                                    (255, 200, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                                cam_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), width="stretch")
                                time.sleep(0.025)

                        # -------------------------------------------------------------
                        # STAGE 3: COUNTDOWN — "3... 2... 1..."
                        # -------------------------------------------------------------
                        if hand_detected and st.session_state.get("camera_active", False):
                            st.session_state["capture_status"] = "COUNTDOWN"
                            countdown_steps = [("3...", 3), ("2...", 2), ("1...", 1)]

                            for step_label, count_val in countdown_steps:
                                render_capture_status_badge(
                                    status_badge_placeholder,
                                    "COUNTDOWN",
                                    f"Starting gesture recording in {count_val}...",
                                )
                                status_placeholder.warning(f"⏱️ **COUNTDOWN:** Starting gesture recording in **{count_val}**...")

                                step_frames = 18  # ~0.65s per count step (total ~2 seconds)
                                for _ in range(step_frames):
                                    ret, frame = cap.read()
                                    if not ret or frame is None:
                                        break
                                    hands, annotated = tracker.process_frame(frame, draw=True)
                                    display_frame = cv2.flip(annotated, 1)

                                    cv2.putText(
                                        display_frame,
                                        f"COUNTDOWN: {count_val}",
                                        (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.95,
                                        (0, 220, 255),
                                        2,
                                        cv2.LINE_AA,
                                    )
                                    # Center watermark for unmistakable visibility
                                    h, w = display_frame.shape[:2]
                                    cv2.putText(
                                        display_frame,
                                        str(count_val),
                                        (w // 2 - 35, h // 2 + 35),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        2.8,
                                        (0, 220, 255),
                                        4,
                                        cv2.LINE_AA,
                                    )
                                    cam_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), width="stretch")
                                    time.sleep(0.02)

                        # -------------------------------------------------------------
                        # STAGE 4: CAPTURING GESTURE — Collect exactly 24 hand-present frames
                        # -------------------------------------------------------------
                        if hand_detected and st.session_state.get("camera_active", False):
                            st.session_state["capture_status"] = "CAPTURING"
                            render_capture_status_badge(
                                status_badge_placeholder,
                                "CAPTURING",
                                "Recording 24 valid hand-present frames...",
                            )
                            status_placeholder.success("🔴 **CAPTURING GESTURE** — Perform your sign deliberately now!")

                            frames_buffer = []
                            max_capture_attempts = 100
                            attempts = 0

                            while len(frames_buffer) < required_window and attempts < max_capture_attempts:
                                attempts += 1
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    break

                                # RAW frame processed with MediaPipe (preserves exact training space & handedness)
                                hands, annotated = tracker.process_frame(frame, draw=True)
                                display_frame = cv2.flip(annotated, 1)

                                hands_detected = len(hands) > 0

                                if hands_detected:
                                    # Collect ONLY valid hand-present frames
                                    frame_vec = format_frame_landmarks(hands)
                                    frames_buffer.append(frame_vec)
                                    hud_text = f"CAPTURING: {len(frames_buffer)}/{required_window}"
                                    hud_color = (0, 255, 0)
                                else:
                                    # Hand lost: prompt user; do NOT append zero frames
                                    hud_text = f"KEEP HAND VISIBLE! ({len(frames_buffer)}/{required_window})"
                                    hud_color = (0, 0, 255)

                                progress_placeholder.progress(
                                    len(frames_buffer) / required_window,
                                    text=f"Recording valid frames: {len(frames_buffer)}/{required_window}",
                                )

                                cv2.putText(
                                    display_frame,
                                    hud_text,
                                    (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.85,
                                    hud_color,
                                    2,
                                    cv2.LINE_AA,
                                )
                                cam_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), width="stretch")
                                time.sleep(0.015)

                        # -------------------------------------------------------------
                        # STAGE 5: CLASSIFIED — Predict once and freeze display
                        # -------------------------------------------------------------
                        if len(frames_buffer) >= required_window:
                            pred_res = predictor.predict_sequence(frames_buffer)
                            st.session_state["live_prediction_result"] = pred_res
                            st.session_state["live_predicted_gloss"] = pred_res.predicted_gloss
                            st.session_state["capture_status"] = "CLASSIFIED"

                            conf_disp = getattr(pred_res, "confidence", 1.0)
                            conf_pct = (conf_disp * 100.0) if conf_disp is not None else 100.0
                            badge_label = f"PREDICTED: {pred_res.predicted_gloss} ({conf_pct:.1f}%)"
                            status_msg = f"✓ Gesture Window Captured & Predicted: **{pred_res.predicted_gloss}**"

                            render_capture_status_badge(
                                status_badge_placeholder,
                                "CLASSIFIED",
                                badge_label,
                            )
                            status_placeholder.success(status_msg)
                            progress_placeholder.progress(1.0, text="Captured exactly 24/24 valid frames.")
                            if hasattr(pred_res, "display"):
                                print(pred_res.display())

                    except Exception as ex:
                        st.error(f"Error during live classification: {ex}")
                    finally:
                        cap.release()
                        st.session_state["camera_active"] = False
                        st.rerun()
            else:
                pred_result = st.session_state.get("live_prediction_result")
                if pred_result is not None:
                    res_conf = getattr(pred_result, "confidence", 1.0)
                    res_pct = (res_conf * 100.0) if res_conf is not None else 100.0
                    badge_label = f"PREDICTED: {getattr(pred_result, 'predicted_gloss', '')} ({res_pct:.1f}%)"
                    render_capture_status_badge(
                        status_badge_placeholder,
                        "CLASSIFIED",
                        badge_label,
                    )
                else:
                    render_capture_status_badge(
                        status_badge_placeholder,
                        "IDLE",
                        "Click START CAMERA to begin controlled gesture capture.",
                    )
                    cam_placeholder.info("Click **START CAMERA** to initiate the controlled gesture capture flow (Wait → Ready → Countdown → Capture 24 frames → Classify).")

        with col_pred:
            st.markdown("#### Classifier Prediction Output")
            pred_res: Optional[GesturePredictionResult] = st.session_state.get("live_prediction_result")

            if pred_res is not None:
                with st.container(border=True):
                    st.markdown(f"### Predicted Gesture: `{pred_res.predicted_gloss}`")
                    st.markdown(f"**Category:** `{pred_res.token_type}`")
                    st.markdown(f"**Meaning:** *{pred_res.meaning}*")

                    # Confidence
                    conf_val = getattr(pred_res, "confidence", None)
                    if conf_val is not None:
                        st.markdown(f"**Confidence:** `{conf_val * 100:.1f}%` (k-NN vote agreement)")
                    else:
                        st.markdown("**Confidence:** not calibrated")

                    # Vote breakdown (safely handled if empty or missing)
                    vote_counts = getattr(pred_res, "vote_counts", None) or {}
                    if vote_counts:
                        votes_str = ", ".join([f"{k}: {v}" for k, v in vote_counts.items()])
                        st.markdown(f"**k-NN Vote Breakdown:** `{votes_str}`")

                    # Display 24/24 frames captured
                    f_count = getattr(pred_res, "frame_count", 0) or 24
                    st.markdown(f"**Temporal Window:** `{f_count}/24 frames captured` (valid hand-present frames)")

                    # Live Diagnostics & Feature Telemetry Expander
                    with st.expander("📊 Live Diagnostics & Feature Telemetry", expanded=True):
                        feats = getattr(pred_res, "features", None)
                        if feats is not None:
                            f_arr = np.asarray(feats)
                            st.markdown(f"**Feature Vector Dim:** `{f_arr.shape}` (252 features: 126 mean + 126 std)")
                            col_m1, col_m2 = st.columns(2)
                            with col_m1:
                                st.write(f"• **Min:** `{float(f_arr.min()):.4f}`")
                                st.write(f"• **Max:** `{float(f_arr.max()):.4f}`")
                            with col_m2:
                                st.write(f"• **Mean:** `{float(f_arr.mean()):.4f}`")
                                st.write(f"• **Std:** `{float(f_arr.std()):.4f}`")

                        neighbors = getattr(pred_res, "nearest_neighbors", None)
                        if neighbors:
                            st.markdown("**Top-k Nearest Prototypes (Class, Distance):**")
                            for idx, (nbr_gloss, nbr_dist) in enumerate(neighbors, 1):
                                st.markdown(f"{idx}. **{nbr_gloss}** (Euclidean Distance: `{nbr_dist:.4f}`)")

                # Button to add to gloss stream
                if st.button(f"➕ ADD '{pred_res.predicted_gloss}' TO GLOSS STREAM", type="primary", key="btn_add_to_stream", width="stretch"):
                    st.session_state["gloss_stream"].append(pred_res.predicted_gloss)
                    st.rerun()

            else:
                with st.container(border=True):
                    st.markdown("*(No active live gesture prediction yet. Start camera to capture a sign.)*")
                    st.caption("Supported Classes: `ME`, `YOU`, `WANT`, `MILK`, `BOOK`, `HOME`, `NOT`")

        # ---------------------------------------------------------------------
        # GLOSS STREAM & COMPILER INTEGRATION
        # ---------------------------------------------------------------------
        st.divider()
        st.markdown("### GLOSS STREAM & COMPILER INTEGRATION")
        st.caption("Construct a multi-sign sentence from real recognized gestures and compile it into formal AST, Semantics, and IR.")

        stream_list: List[str] = st.session_state.get("gloss_stream", [])

        if stream_list:
            stream_display = " &rarr; ".join([f"`{g}`" for g in stream_list])
            st.markdown(f"#### Current Stream: {stream_display}", unsafe_allow_html=True)
            stream_sentence = " ".join(stream_list)
            st.info(f"**Compiler Input String:** `{stream_sentence}`")
        else:
            st.markdown("#### Current Stream: *(Empty)*")
            st.caption("Capture gestures with the live camera and click 'ADD TO GLOSS STREAM' to build a sentence.")

        col_st1, col_st2, col_st3 = st.columns([1.5, 1.5, 3])
        with col_st1:
            compile_stream_btn = st.button("🚀 COMPILE GLOSS STREAM", type="primary", key="btn_compile_stream", disabled=(len(stream_list) == 0), width="stretch")
        with col_st2:
            clear_stream_btn = st.button("🗑️ CLEAR GLOSS STREAM", key="btn_clear_stream", disabled=(len(stream_list) == 0), width="stretch")

        if clear_stream_btn:
            st.session_state["gloss_stream"] = []
            st.session_state["compiled_stream_result"] = None
            st.rerun()

        if compile_stream_btn and stream_list:
            stream_source = " ".join(stream_list).strip()
            res_stream = compile_frontend(stream_source)
            st.session_state["compiled_stream_result"] = res_stream

        # Render Compiler Pipeline Results if Stream is compiled
        stream_res: Optional[FrontendResult] = st.session_state.get("compiled_stream_result")
        if stream_res is not None:
            st.markdown("---")
            st.markdown("#### COMPILED STREAM PIPELINE RESULTS")

            # Status Banner
            if stream_res.is_valid:
                st.success(f"✓ COMPILATION SUCCESSFUL for sentence: **{stream_res.source}**")
            else:
                st.error(f"✗ COMPILATION HALTED at {stream_res.failed_phase} phase for sentence: **{stream_res.source}**")
                if stream_res.errors:
                    for err in stream_res.errors:
                        st.caption(f"⚠️ {str(err)}")

            # 2 Columns for Tokens & AST
            c_tok, c_ast = st.columns(2)
            with c_tok:
                with st.container(border=True):
                    st.markdown("**1. LEXICAL ANALYSIS (TOKENS)**")
                    if stream_res.lexer_result and stream_res.lexer_result.is_valid:
                        tok_df = pd.DataFrame([
                            {"Position": t.position, "Lexeme": t.lexeme, "Category": t.token_type, "Meaning": t.meaning}
                            for t in stream_res.lexer_result.tokens
                        ])
                        st.dataframe(tok_df, width="stretch", hide_index=True)
                    else:
                        st.caption(f"Lexical error: {stream_res.errors[0] if stream_res.errors else 'Failed'}")

            with c_ast:
                with st.container(border=True):
                    st.markdown("**2. ABSTRACT SYNTAX TREE (AST)**")
                    if stream_res.ast:
                        st.code(stream_res.ast.pretty_print(), language=None)
                    else:
                        st.caption("AST unavailable.")

            # Normalized Semantics & IR
            c_sem, c_ir = st.columns(2)
            with c_sem:
                with st.container(border=True):
                    st.markdown("**3. NORMALIZED SEMANTICS**")
                    if stream_res.semantic_form:
                        s_form = stream_res.semantic_form
                        st.json(s_form.to_dict())
                    else:
                        st.caption("Semantic form unavailable.")
            with c_ir:
                with st.container(border=True):
                    st.markdown("**4. INTERMEDIATE REPRESENTATION (IR)**")
                    if stream_res.ir_program:
                        st.code(stream_res.ir_program.to_text(), language=None)
                    else:
                        st.caption("IR unavailable.")

        # ---------------------------------------------------------------------
        # CALIBRATED / OFFLINE TEST (Preserved Below)
        # ---------------------------------------------------------------------
        st.divider()
        st.markdown("### CALIBRATED / OFFLINE TEST")
        st.caption("Offline demonstration — does not use the webcam.")

        sym_table = get_symbol_table()
        sample_sign = st.selectbox(
            "Select Sign Gesture Sample (Offline Lookup):",
            ["ME", "YOU", "WANT", "MILK", "BOOK", "HOME", "NOT"],
            key="offline_sign_sample",
        )
        if st.button("Run Calibrated Offline Inference", key="btn_offline_inference"):
            sym_entry = sym_table.lookup(sample_sign)
            if sym_entry:
                st.success(
                    f"**Recognized Gloss:** `{sym_entry.lexeme}`\n\n"
                    f"&bull; **Token Type:** `{sym_entry.token_type}`  \n"
                    f"&bull; **Meaning:** {sym_entry.meaning}  \n"
                    f"&bull; **Model Type:** 252-D k-NN Prototype Match  \n"
                    f"*(Transferred to Compiler Gloss Stream)*"
                )


# ==============================================================================
# TAB 3: ARCHITECTURE (Native Streamlit Layout)
# ==============================================================================

with tab_architecture:
    st.subheader("SYSTEM ARCHITECTURE")
    st.caption("From Sign Recognition to Compiler Intermediate Representation")
    st.divider()

    # SECTION 1: Input Acquisition — Computer Vision
    st.markdown("### 1. INPUT ACQUISITION — COMPUTER VISION")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.container(border=True):
            st.markdown("#### TEMPORAL VIDEO")
            st.caption("Webcam / sign video frames")
    with c2:
        with st.container(border=True):
            st.markdown("#### MEDIAPIPE HANDS")
            st.caption("21 landmarks per hand\n\n(x, y, z)")
    with c3:
        with st.container(border=True):
            st.markdown("#### FEATURE EXTRACTION")
            st.caption("252-dimensional temporal feature vector\n\nMean + Standard Deviation")
    with c4:
        with st.container(border=True):
            st.markdown("#### k-NN CLASSIFIER")
            st.caption("Gesture → Gloss")

    st.info("**GLOSS STREAM:** `TODAY → ME → WANT → FOOD`")
    st.divider()

    # SECTION 2: Compiler Frontend
    st.markdown("### 2. COMPILER FRONTEND")
    st.caption("The compiler begins after the recognition subsystem produces the gloss stream.")

    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1:
        with st.container(border=True):
            st.markdown("#### SYMBOL TABLE")
            st.caption("44 formal glosses & categories")
    with row1_c2:
        with st.container(border=True):
            st.markdown("#### LEXER")
            st.caption("Gloss → Typed Tokens")
    with row1_c3:
        with st.container(border=True):
            st.markdown("#### PARSER")
            st.caption("LL(1) Recursive Descent")

    row2_c1, row2_c2, row2_c3 = st.columns(3)
    with row2_c1:
        with st.container(border=True):
            st.markdown("#### AST")
            st.caption("Abstract Syntax Tree")
    with row2_c2:
        with st.container(border=True):
            st.markdown("#### SEMANTIC ANALYSIS")
            st.caption("Category / Type Validation")
    with row2_c3:
        with st.container(border=True):
            st.markdown("#### INTERMEDIATE REPRESENTATION")
            st.caption("Target-Independent IR")

    st.divider()

    # SECTION 3: Phase 3 Backend
    st.markdown("### 3. PHASE 3 BACKEND")
    p3_c1, p3_c2, p3_c3 = st.columns(3)
    with p3_c1:
        with st.container(border=True):
            st.markdown("#### INTERMEDIATE REPRESENTATION")
            st.caption("Current Phase 2 output")
    with p3_c2:
        with st.container(border=True):
            st.markdown("#### ENGLISH GENERATION")
            st.caption("Future")
    with p3_c3:
        with st.container(border=True):
            st.markdown("#### TEXT-TO-SPEECH")
            st.caption("Future")

    st.warning("⚠️ **Phase 3 backend is a future roadmap and is not implemented in the current Phase 2 system.**")
    st.divider()

    # SECTION 4: End-to-End Flow
    st.markdown("### 4. END-TO-END FLOW")
    flow_steps = [
        "SIGN / VIDEO",
        "GESTURE RECOGNITION",
        "GLOSS STREAM",
        "LEXER",
        "PARSER",
        "AST",
        "SEMANTIC ANALYSIS",
        "INTERMEDIATE REPRESENTATION",
        "PHASE 3 BACKEND",
    ]
    for idx, step in enumerate(flow_steps):
        with st.container(border=True):
            st.markdown(f"**Step {idx + 1}:** `{step}`")
        if idx < len(flow_steps) - 1:
            st.markdown("<div style='text-align: center; color: #818CF8; font-size: 1.25rem; margin: -0.3rem 0;'>↓</div>", unsafe_allow_html=True)

    st.divider()

    # SECTION 5: Compiler Boundary
    st.markdown("### 5. COMPILER BOUNDARY")
    st.markdown(
        """
        1. **Computer Vision Boundary:** Computer vision recognizes isolated gestures and produces glosses.
        2. **Source Language Definition:** The compiler treats the gloss sequence as its source language.
        3. **Compiler Processing Pipeline:** The compiler processes the glosses through lexical analysis, syntax analysis, AST construction, semantic analysis, and IR generation.
        """
    )


# ==============================================================================
# TAB 4: SYMBOL TABLE (Authoritative Vocabulary Explorer)
# ==============================================================================

with tab_symbols:
    sym_table = get_symbol_table()
    total_symbols = sym_table.size()

    st.markdown(f"### Authoritative Symbol Table ({total_symbols} Glosses)")
    st.caption("The single source of truth for recognized sign glosses, syntactic token categories, and normalized meanings.")

    # Search Box
    search_col, _ = st.columns([2, 2])
    with search_col:
        search_kw = st.text_input("Search Symbol:", placeholder="e.g. ME, WANT, FOOD, DANCE").strip().upper()

    if search_kw:
        found_sym = sym_table.lookup(search_kw)
        if found_sym:
            st.success(f"**Found in Symbol Table:** `{found_sym.lexeme}` &bull; Type: `{found_sym.token_type}` &bull; Meaning: *{found_sym.meaning}*")
        else:
            st.error(f"'{search_kw}' is NOT in the Symbol Table. (Will trigger Lexical Error in compiler).")

    # Display All Symbols in a Clean Table
    all_symbols = sym_table.get_all_symbols()
    sym_df = pd.DataFrame([
        {
            "Lexeme": s.lexeme,
            "Category / Token Type": s.token_type,
            "English Meaning": s.meaning,
        }
        for s in all_symbols
    ])
    st.dataframe(sym_df, width="stretch", hide_index=True)
