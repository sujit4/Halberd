"""
Hypercorn server configuration and management module for the Halberd Multi-Cloud Attack Tool.

This module provides server configuration and deployment capabilities for Halberd using Hypercorn ASGI server.
It supports both development and production deployments with configurable SSL, logging, and network settings.

Classes:
    Server: Main server configuration class that manages Hypercorn server settings and startup.

Functions:
    main(): Command-line interface for server configuration and startup.

Environment Variables:
    HALBERD_HOST: Host address to bind the server to (default: 127.0.0.1)
    HALBERD_PORT: Port number to run the server on (default: 8050)

Command Line Arguments:
    --host: Host address to bind to (overrides HALBERD_HOST)
    --port: Port to bind to (overrides HALBERD_PORT)
    --ssl-cert: Path to SSL certificate file for HTTPS
    --ssl-key: Path to SSL private key file for HTTPS
    --log-level: Server logging level (debug/info/warning/error/critical)
    --dev-server: Flag to use Flask development server instead of Hypercorn
    --dev-server-debug: Enable debug mode for development server

Example Usage:
    # Start production server
    python server.py

    # Start with custom host and port
    python server.py --host 0.0.0.0 --port 8443

    # Start with SSL
    python server.py --ssl-cert cert.pem --ssl-key key.pem

    # Start development server
    python server.py --dev-server

Notes:
    - The Server class validates SSL configurations and port numbers
    - Production deployments should use Hypercorn (default) instead of the development server
    - SSL certificate and key must be provided together for HTTPS
    - Log files are written to the configured server log file path
"""

import argparse
import asyncio
import os
from datetime import datetime

from hypercorn.asyncio import serve
from hypercorn.config import Config

from core.bootstrap import Bootstrapper
from version import __version__


class Server:
    """Halberd Server Configuration"""

    def __init__(
        self,
        host="127.0.0.1",
        port=8050,
        ssl_cert=None,
        ssl_key=None,
        log_level="warning",
        server_log_file="./local/server.log",
    ):
        # Port number validation
        if not (0 <= port <= 65535):
            raise ValueError(f"Port number must be between 0 and 65535, got {port}")
        self.host = host
        self.port = port
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.app = None
        self.server_log_file = server_log_file
        self.log_level = log_level

    def _validate_ssl(self):
        """Validate SSL certificate and key if provided"""
        if bool(self.ssl_cert) != bool(self.ssl_key):
            raise ValueError("Both SSL certificate and key must be provided together")
        if self.ssl_cert and not os.path.exists(self.ssl_cert):
            raise ValueError(f"SSL certificate not found: {self.ssl_cert}")
        if self.ssl_key and not os.path.exists(self.ssl_key):
            raise ValueError(f"SSL key not found: {self.ssl_key}")

    def _get_hypercorn_config(self):
        """Generate Hypercorn configuration"""
        config = Config()
        config.bind = [f"{self.host}:{self.port}"]

        if self.ssl_cert and self.ssl_key:
            config.certfile = self.ssl_cert
            config.keyfile = self.ssl_key

        # Additional settings
        config.loglevel = self.log_level  # Defaults to 'warning'
        config.accesslog = self.server_log_file  # Log to server log file
        config.errorlog = self.server_log_file  # Log to server log file
        config.worker_class = "asyncio"
        config.keep_alive_timeout = 65

        return config

    async def run(self):
        """Start Halberd server"""
        try:
            # SSL check
            self._validate_ssl()

            if self.app is None:
                # Import Halberd application
                from halberd import app

                self.app = app.server

            # Log server configuration
            protocol = "https" if self.ssl_cert else "http"
            url = f"{protocol}://{self.host}:{self.port}"

            # Display startup info
            print_banner()
            print_startup_info(self.host, self.port, url)

            # Start Hypercorn
            config = self._get_hypercorn_config()
            await serve(self.app, config)

        except Exception as e:
            raise RuntimeError(f"Server startup failed: {str(e)}")


def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                          HALBERD                             ║
║               Multi-Cloud Agentic Attack Tool                ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_startup_info(host, port, url):
    """Print startup information"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(">> Application Status:")
    print(f"   └─ Started at: {timestamp}")
    print(f"   └─ Version: {__version__}")
    print()

    print(">> Server Configuration:")
    print(f"   └─ Host: {host}")
    print(f"   └─ Port: {port}")
    print(f"   └─ URL: {url}")
    print()

    print("[Success] Initialization complete")
    print("=" * 60)


def main():
    """Command line interface for the server with defaults and environment variable support"""
    parser = argparse.ArgumentParser(
        description="Halberd Multi-Cloud Attack Tool Server"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HALBERD_HOST", "127.0.0.1"),
        help="Host address to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("HALBERD_PORT", "8050")),
        help="Port to bind to",
    )
    parser.add_argument("--ssl-cert", help="Path to SSL certificate file")
    parser.add_argument("--ssl-key", help="Path to SSL key file")
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Server logging level",
    )
    parser.add_argument(
        "--dev-server",
        action="store_true",
        help="Flag launches Flask development server instead of Hypercorn",
    )
    parser.add_argument(
        "--dev-server-debug",
        action="store_true",
        help="Flag enables debug mode for development server",
    )
    args = parser.parse_args()

    # Initialize application requirements
    bootstrapper = Bootstrapper()
    bootstrapper.initialize()

    if args.dev_server:
        # Start development server
        from halberd import app

        app.run(
            host=args.host or "127.0.0.1",
            port=args.port or "8050",
            debug=args.dev_server_debug or False,
        )
    else:
        # Start production hypercorn server
        server = Server(
            host=args.host, port=args.port, ssl_cert=args.ssl_cert, ssl_key=args.ssl_key
        )

        asyncio.run(server.run())  # Run the async server


if __name__ == "__main__":
    main()
