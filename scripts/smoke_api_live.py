"""Launch the real API and exercise HTTP success and validation-failure paths."""
import socket
import subprocess
import sys
import time

import httpx


def main():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "financial_risk.api.app:app",
                                "--host", "127.0.0.1", "--port", str(port)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False) as client:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError("API process exited before readiness")
                try:
                    if client.get("/ready").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("API did not become ready within 30 seconds")
                time.sleep(.2)
            signals = {"fraud_probability": .9, "anomaly_score": .8,
                       "network_score": .7, "velocity_score": .6}
            score = client.post("/v1/risk/score", json=signals)
            assert score.status_code == 200 and abs(score.json()["risk_score"] - .81) < 1e-9
            case = client.post("/v1/investigations/cases", json={**signals,
                "transaction": {"transaction_id": "SYNTHETIC-DEMO", "amount": 250}})
            assert case.status_code == 200 and case.json()["transaction_id"] == "SYNTHETIC-DEMO"
            assert client.post("/v1/risk/score", json={**signals, "fraud_probability": 2}).status_code == 422
        print("Live API smoke passed: readiness, scoring, case creation, invalid-input rejection")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
