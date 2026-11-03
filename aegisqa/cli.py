"""
Command-Line Interface (CLI) for AegisQA Enterprise Platform.
Usage:
  python3 -m aegisqa.cli gate       # Run full 21-test CI quality gate & generate JSON report
  python3 -m aegisqa.cli serve      # Start live TCP HTTP Gateway on port 8080
  python3 -m aegisqa.cli load-test  # Run concurrent TCP socket load test
  python3 -m aegisqa.cli drift      # Compute PSI & KL Divergence across SQL telemetry warehouse
"""
import argparse
import json
import sys
from aegisqa.server.http_gateway import OmniHomeAPIGatewaySimulator
from aegisqa.pipeline.statistical_drift import TelemetryDataPipelineValidator


def main() -> None:
    parser = argparse.ArgumentParser(prog="aegisqa", description="AegisQA Enterprise AI & IoT Quality CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gate", help="Run full CI/CD quality gate and generate reports/latest_eval_report.json")
    serve_p = sub.add_parser("serve", help="Start live multi-threaded OpenAPI 3.1 HTTP server")
    serve_p.add_argument("--port", type=int, default=8080)
    load_p = sub.add_parser("load-test", help="Run concurrent TCP socket load benchmark")
    load_p.add_argument("--requests", type=int, default=60)
    load_p.add_argument("--concurrency", type=int, default=10)
    sub.add_parser("drift", help="Run SQLite/Athena PSI & KL-Divergence statistical drift audit")

    args = parser.parse_args()
    if args.command == "gate":
        from run_quality_gate import generate_enterprise_report
        generate_enterprise_report()
    elif args.command == "serve":
        gw = OmniHomeAPIGatewaySimulator()
        srv, url = gw.start_live_http_server(host="0.0.0.0", port=args.port)
        print(f"AegisQA HTTP Gateway listening at {url} (Ctrl+C to stop)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            srv.shutdown()
    elif args.command == "load-test":
        gw = OmniHomeAPIGatewaySimulator()
        res = gw.run_load_test(total_requests=args.requests, concurrency=args.concurrency)
        print(json.dumps(res, indent=2))
    elif args.command == "drift":
        val = TelemetryDataPipelineValidator()
        print(json.dumps(val.compute_distribution_drift_report(), indent=2))


if __name__ == "__main__":
    main()
