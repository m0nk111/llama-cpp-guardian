"""PR-Piet end-to-end test: kleine, veilige wijziging.

- Voegt een helper toe die door bestaande code gebruikt kan worden
- Bevat een subtiel probleem (lege except) zodat de reviewer iets kan vinden
- Bevat een bestaande-taal-wijziging (Python) zodat de tree-sitter mapper
  symbolen en callers kan tonen
"""

from __future__ import annotations


def parse_port(raw: str, default: int = 8080) -> int:
    """Parse een poort uit een string; ongeldige input -> default."""
    try:
        return int(raw)
    except ValueError:
        return default
    except TypeError:
        return default
    except Exception:  # noqa: BLE001 - bewust breed: alle fouten -> default
        return default


def is_loopback(host: str) -> bool:
    """True voor loopback-hosts (localhost, 127.0.0.1, ::1)."""
    return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("127.")
