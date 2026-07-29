"""Streamlit UI for the multi-user document Q&A demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from graph import build_graph, create_initial_state, run_graph

USER_DB_PATH = ROOT / "users.json"


@st.cache_resource
def load_graph() -> Any:
    return build_graph()


def load_users() -> dict[str, list[str]]:
    with USER_DB_PATH.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return {email: companies for email, companies in loaded.items()}


def init_session_state() -> None:
    if "logged_in_user" not in st.session_state:
        st.session_state.logged_in_user = None
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if "thread_ids" not in st.session_state:
        st.session_state.thread_ids = {}


def main() -> None:
    st.set_page_config(page_title="Multi-user Document Q&A", page_icon="📄", layout="wide")
    init_session_state()

    st.title("Multi-user earnings-call Q&A")
    st.caption("Login as a demo user, ask questions, and keep follow-up context isolated per user.")

    users = load_users()

    with st.sidebar:
        st.header("Login")
        email = st.text_input("Email", placeholder="alice@email.com")
        login_error = ""
        if st.button("Log in", use_container_width=True):
            if not email:
                st.warning("Enter an email to continue.")
            elif email not in users:
                st.error("Unknown email. Use one of the demo users from users.json.")
            else:
                st.session_state.logged_in_user = email
                if email not in st.session_state.chat_histories:
                    st.session_state.chat_histories[email] = []
                st.session_state.thread_ids[email] = f"thread::{email}"
                st.rerun()

        if st.session_state.logged_in_user:
            st.success(f"Signed in as {st.session_state.logged_in_user}")
            st.write("Authorized companies:")
            for company in users[st.session_state.logged_in_user]:
                st.write(f"- {company}")

            if st.button("Log out / switch user", use_container_width=True):
                st.session_state.logged_in_user = None
                st.rerun()

        st.divider()
        st.subheader("Demo users")
        for user_email, companies in users.items():
            st.write(f"- {user_email}: {', '.join(companies)}")

    if not st.session_state.logged_in_user:
        st.info("Enter a known email in the sidebar to start a session.")
        return

    user_email = st.session_state.logged_in_user
    allowed_companies = users[user_email]

    history = st.session_state.chat_histories.get(user_email, [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your authorized documents...")
    if question:
        graph = load_graph()
        state = create_initial_state(
            user_email=user_email,
            allowed_companies=allowed_companies,
            messages=history,
        )
        state["current_question"] = question
        thread_id = st.session_state.thread_ids.get(user_email, f"thread::{user_email}")

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching authorized documents..."):
                result = run_graph(graph, thread_id, state)
            st.markdown(result["answer"])

        history = result["messages"]
        st.session_state.chat_histories[user_email] = history


if __name__ == "__main__":
    main()
