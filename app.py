"""
app.py
Main entry point for Auto_Fine_Tuning — SFT · DPO · Auto-ORPO.
Usage: python app.py [--port 7860] [--no-browser]
"""

import argparse
import socket
import sys
import os
import threading
import webbrowser


def _get_lan_ip() -> str:
    """Return the LAN IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Auto_Fine_Tuning — SFT · DPO · Auto-ORPO")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0 for LAN+localhost; use 127.0.0.1 for localhost only)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    # Ensure the working directory is the project root.
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Check whether Gradio is installed.
    try:
        import gradio
    except ImportError:
        print("❌ Gradio is not installed. Run: pip install 'gradio>=6.0'")
        sys.exit(1)

    print("=" * 60)
    print("  Auto_Fine_Tuning — SFT · DPO · Auto-ORPO")
    print("=" * 60)
    print("Initializing environment detection...")

    from ui.app_builder import build_app
    app, launch_kwargs = build_app()

    lan_ip = _get_lan_ip()
    localhost_url = f"http://localhost:{args.port}"
    lan_url = f"http://{lan_ip}:{args.port}"

    print("\nStarting Gradio server...")
    print(f"  Local:   {localhost_url}")
    if args.host == "0.0.0.0" and lan_ip != "unknown":
        print(f"  Network: {lan_url}")
    print("Press Ctrl+C to exit.\n")

    # Open browser to localhost after server starts (not 0.0.0.0)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(localhost_url)).start()

    # Bypass system proxy for localhost — Gradio 6 startup check hits localhost
    # and macOS system proxies can intercept it causing a 502 error.
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

    app.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=False,   # we handle browser opening manually above
        share=args.share,
        show_error=True,
        quiet=False,
        **launch_kwargs,
    )


if __name__ == "__main__":
    main()
