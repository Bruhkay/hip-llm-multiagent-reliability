# %% [code]
# %% [code]
# %% [code]
from __future__ import annotations

UTILS_VERSION = "2026-08-07_v8_content_gated_capture"

import asyncio
import base64
import copy
import importlib
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlparse

import requests
import websockets
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# PixelRAG remains the renderer used by the workshop.
PIXELRAG_REPO = Path("/kaggle/working/PixelRAG")
PIXELRAG_RENDER_SRC = PIXELRAG_REPO / "render" / "src"
if not PIXELRAG_RENDER_SRC.exists():
    raise FileNotFoundError(
        "PixelRAG was not found. Run the notebook installation cell first: "
        f"{PIXELRAG_RENDER_SRC}"
    )

render_src_text = str(PIXELRAG_RENDER_SRC)
if render_src_text not in sys.path:
    sys.path.insert(0, render_src_text)
importlib.invalidate_caches()

try:
    from pixelrag_render import render_url
except ImportError:
    from pixelrag_render.render import render_url
from pixelrag_render.chrome import find_chrome
from pixelrag_render.backends.cdp import BROWSER_ARGS

OUTPUT_DIR = Path("/kaggle/working")
CAPTURE_ROOT = OUTPUT_DIR / "echa_pixelrag_capture"
GRAPH_MERMAID_PATH = OUTPUT_DIR / "smiles_echa_pixelrag_graph.mmd"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "smiles_echa_pixelrag_report.txt"
JSON_OUTPUT_PATH = OUTPUT_DIR / "smiles_echa_pixelrag_final.json"
CHROME_STDERR_PATH = OUTPUT_DIR / "pixelrag_chrome_stderr.log"
CDP_SESSION_PATH = OUTPUT_DIR / "pixelrag_cdp_session.json"
CONSENT_BEFORE_PATH = OUTPUT_DIR / "echa_consent_before.png"
CONSENT_AFTER_PATH = OUTPUT_DIR / "echa_consent_after.png"
PRE_PIXELRAG_PATH = OUTPUT_DIR / "echa_pre_pixelrag.png"
PREPARED_MIRROR_PATH = OUTPUT_DIR / "echa_prepared_page.html"

MODEL = os.environ.get("OPENAI_AGENT_MODEL", "gpt-5-mini")
CDP_PORT = 9222
MAX_VISION_TILES = 8
REQUEST_TIMEOUT = 45

# ECHA CHEM is an Angular application: the shell (header, side navigation and the
# "Industry classifications" heading) paints long before the CLP notification data
# arrives. Every wait in this module therefore polls for the hazard content itself
# instead of sleeping for a fixed number of seconds.
# A reload recovers a shell-only response far more reliably than waiting longer,
# so the per-attempt budget is short and the attempt count carries the patience.
# Worst case here is 3 x (3 + 3 + 45) = ~153s, which fits the worker timeouts below.
CONTENT_TIMEOUT_SECONDS = 45.0
MAX_PREPARE_ATTEMPTS = 3

# PixelRAG's CDP backend calls Target.createTarget, so it always captures in its own
# fresh tab and stops at "network idle". For this SPA network idle can be reached in
# the gap between bootstrap and the data request, which yields a tile whose content
# column is blank. The capture agent therefore scores each tile and retries.
MAX_LIVE_RENDER_ATTEMPTS = 3
TILE_CONTENT_ROW_THRESHOLD = 0.10


def load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("OPENAI_API_KEY").strip()
    except Exception:
        return ""


OPENAI_API_KEY = load_openai_key()
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. Add it to Kaggle Secrets or the environment."
    )

client = OpenAI(api_key=OPENAI_API_KEY, timeout=240.0, max_retries=2)
http = requests.Session()
retry_policy = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
)
http.mount("https://", HTTPAdapter(max_retries=retry_policy))

# Type alias used only for annotations in helper functions.
AgentState = dict[str, Any]


def audit_event(agent: str, action: str, status: str, details: str) -> list[dict[str, Any]]:
    return [{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": agent,
        "action": action,
        "status": status,
        "details": details,
    }]


STRING = {"type": "string"}
STRING_LIST = {"type": "array", "items": STRING}


def strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


ECHA_URL_SCHEMA = strict_object({
    "status": {"type": "string", "enum": ["found", "not_found"]},
    "matched_name": STRING,
    "matched_cas": STRING,
    "matched_ec_number": STRING,
    "substance_url": STRING,
    "industry_url": STRING,
    "source_urls": STRING_LIST,
    "notes": STRING,
})



CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")
HAZARD_CODE_PATTERN = re.compile(r"^(?:H\d{3}[A-Z]*|EUH\d{3})$", re.IGNORECASE)


def valid_cas(value: str) -> bool:
    value = str(value or "").strip()
    if not CAS_PATTERN.fullmatch(value):
        return False

    digits = value.replace("-", "")
    check_digit = int(digits[-1])
    weighted = sum(
        int(digit) * weight
        for weight, digit in enumerate(reversed(digits[:-1]), start=1)
    )
    return weighted % 10 == check_digit


def is_echa_url(url: str) -> bool:
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return hostname == "echa.europa.eu" or hostname.endswith(".echa.europa.eu")


def collect_urls(value: Any) -> set[str]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()

    found: set[str] = set()

    if isinstance(value, dict):
        for key, item in value.items():
            if key == "url" and isinstance(item, str) and item.startswith("http"):
                found.add(item)
            found.update(collect_urls(item))

    elif isinstance(value, list):
        for item in value:
            found.update(collect_urls(item))

    return found


def structured_response(
    *,
    instructions: str,
    content: list[dict[str, Any]],
    schema_name: str,
    schema: dict[str, Any],
    use_web_search: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    request: dict[str, Any] = {
        "model": MODEL,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 2500,
        "store": False,
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    }

    if MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
        request["reasoning"] = {"effort": "low"}

    if use_web_search:
        request["tools"] = [
            {
                "type": "web_search",
                "filters": {
                    "allowed_domains": [
                        "echa.europa.eu",
                        "chem.echa.europa.eu",
                    ]
                },
                "search_context_size": "high",
            }
        ]
        request["tool_choice"] = "required"
        request["include"] = ["web_search_call.action.sources"]

    response = client.responses.create(**request)
    if not response.output_text:
        raise RuntimeError("The response contained no structured output.")

    parsed = json.loads(response.output_text)
    urls = sorted(url for url in collect_urls(response) if is_echa_url(url))
    return parsed, urls


def encode_image(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def find_free_port(preferred: int = 9222) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

def cdp_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


async def raw_cdp_send(
    websocket: Any,
    message_counter: list[int],
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message_counter[0] += 1
    message_id = message_counter[0]

    message: dict[str, Any] = {
        "id": message_id,
        "method": method,
    }
    if params:
        message["params"] = params

    await websocket.send(json.dumps(message))

    while True:
        reply = json.loads(
            await asyncio.wait_for(websocket.recv(), timeout=180)
        )
        if reply.get("id") == message_id:
            if "error" in reply:
                raise RuntimeError(
                    f"CDP error for {method}: {reply['error']}"
                )
            return reply.get("result", {})


async def wait_for_cdp_page(
    cdp_url: str,
    *,
    timeout_seconds: float = 30.0,
) -> str:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            targets = cdp_json(f"{cdp_url}/json")
            page_targets = [
                item
                for item in targets
                if item.get("type") == "page"
                and item.get("webSocketDebuggerUrl")
            ]
            if page_targets:
                return page_targets[0]["webSocketDebuggerUrl"]
        except Exception:
            pass

        await asyncio.sleep(0.5)

    raise RuntimeError(
        f"No CDP page target became available at {cdp_url}."
    )



CONSENT_SCROLL_SCRIPT = r"""
(() => {
  const visitedRoots = new Set();
  const scrollables = [];
  const controls = [];

  const normalise = (value) => (
    value || ''
  ).replace(/\s+/g, ' ').trim();

  const visitRoot = (root, offsetX = 0, offsetY = 0, label = 'document') => {
    if (!root || visitedRoots.has(root)) return;
    visitedRoots.add(root);

    const elements = Array.from(root.querySelectorAll('*'));

    for (const element of elements) {
      try {
        if (
          element.scrollHeight > element.clientHeight + 20
          && element.clientHeight > 80
        ) {
          element.scrollTop = element.scrollHeight;
          element.dispatchEvent(
            new Event('scroll', { bubbles: true })
          );
          scrollables.push({
            label,
            tag: element.tagName,
            id: element.id || '',
            className: String(element.className || '').slice(0, 180),
            scrollHeight: element.scrollHeight,
            clientHeight: element.clientHeight
          });
        }
      } catch (_) {}

      const tag = (element.tagName || '').toLowerCase();
      if (
        tag === 'button'
        || tag === 'a'
        || (tag === 'input' && ['button', 'submit'].includes(
          (element.type || '').toLowerCase()
        ))
      ) {
        const text = normalise(
          element.innerText
          || element.value
          || element.textContent
        );
        if (text) {
          controls.push({
            text,
            tag,
            disabled: Boolean(element.disabled),
            ariaDisabled: element.getAttribute('aria-disabled') || '',
            label
          });
        }
      }

      if (element.shadowRoot) {
        visitRoot(
          element.shadowRoot,
          offsetX,
          offsetY,
          `${label} > ${element.tagName.toLowerCase()}#shadow-root`
        );
      }

      if (tag === 'iframe') {
        try {
          const frameRect = element.getBoundingClientRect();
          visitRoot(
            element.contentDocument,
            offsetX + frameRect.left,
            offsetY + frameRect.top,
            `${label} > iframe`
          );
        } catch (_) {}
      }
    }
  };

  visitRoot(document);
  window.scrollTo(0, document.documentElement.scrollHeight);

  return {
    scrollableCount: scrollables.length,
    scrollables: scrollables.slice(0, 60),
    controls: controls.slice(0, 100)
  };
})()
"""


CONSENT_FIND_SCRIPT = r"""
(() => {
  const patterns = [
    /^I Accept the terms$/i,
    /I Accept the terms/i,
    /^Accept the terms$/i,
    /^Accept$/i
  ];

  const visitedRoots = new Set();
  const candidates = [];

  const normalise = (value) => (
    value || ''
  ).replace(/\s+/g, ' ').trim();

  const visitRoot = (
    root,
    offsetX = 0,
    offsetY = 0,
    label = 'document'
  ) => {
    if (!root || visitedRoots.has(root)) return;
    visitedRoots.add(root);

    const selector = [
      'button',
      'a',
      'input[type="button"]',
      'input[type="submit"]',
      '[role="button"]'
    ].join(',');

    for (const element of Array.from(root.querySelectorAll(selector))) {
      const text = normalise(
        element.innerText
        || element.value
        || element.textContent
        || element.getAttribute('aria-label')
      );

      if (!patterns.some((pattern) => pattern.test(text))) {
        continue;
      }

      const rect = element.getBoundingClientRect();
      const style = getComputedStyle(element);

      candidates.push({
        text,
        tag: (element.tagName || '').toLowerCase(),
        disabled: Boolean(element.disabled),
        ariaDisabled: element.getAttribute('aria-disabled') || '',
        visible: (
          rect.width > 1
          && rect.height > 1
          && style.display !== 'none'
          && style.visibility !== 'hidden'
          && Number(style.opacity || 1) > 0
        ),
        x: offsetX + rect.left + rect.width / 2,
        y: offsetY + rect.top + rect.height / 2,
        width: rect.width,
        height: rect.height,
        label
      });
    }

    for (const element of Array.from(root.querySelectorAll('*'))) {
      if (element.shadowRoot) {
        visitRoot(
          element.shadowRoot,
          offsetX,
          offsetY,
          `${label} > ${element.tagName.toLowerCase()}#shadow-root`
        );
      }

      if ((element.tagName || '').toLowerCase() === 'iframe') {
        try {
          const frameRect = element.getBoundingClientRect();
          visitRoot(
            element.contentDocument,
            offsetX + frameRect.left,
            offsetY + frameRect.top,
            `${label} > iframe`
          );
        } catch (_) {}
      }
    }
  };

  visitRoot(document);

  candidates.sort((a, b) => {
    const exactA = /^I Accept the terms$/i.test(a.text) ? 0 : 1;
    const exactB = /^I Accept the terms$/i.test(b.text) ? 0 : 1;
    if (exactA !== exactB) return exactA - exactB;
    if (a.visible !== b.visible) return a.visible ? -1 : 1;
    return 0;
  });

  return {
    found: candidates.length > 0,
    candidate: candidates[0] || null,
    candidates
  };
})()
"""


CONSENT_VERIFY_SCRIPT = r"""
(() => {
  const bodyText = (
    document.body?.innerText || ''
  ).replace(/\s+/g, ' ').trim();

  const hasLegalNotice = (
    /Legal notice/i.test(bodyText)
    && /I Accept the terms/i.test(bodyText)
  );

  return {
    hasLegalNotice,
    bodyExcerpt: bodyText.slice(0, 2500),
    url: location.href,
    title: document.title
  };
})()
"""


async def save_cdp_screenshot(
    websocket: Any,
    counter: list[int],
    output_path: Path,
) -> None:
    result = await raw_cdp_send(
        websocket,
        counter,
        "Page.captureScreenshot",
        {
            "format": "png",
            "fromSurface": True,
            "captureBeyondViewport": False,
        },
    )

    image_data = result.get("data", "")
    if image_data:
        output_path.write_bytes(
            base64.b64decode(image_data)
        )



COOKIE_FIND_SCRIPT = r"""
(() => {
  const patterns = [
    /^Accept only essential cookies$/i,
    /^Accept all cookies$/i,
    /Accept only essential cookies/i,
    /Accept all cookies/i
  ];
  const visited = new Set();
  const candidates = [];
  const normalise = (v) => (v || '').replace(/\s+/g, ' ').trim();

  const walk = (root, ox = 0, oy = 0, label = 'document') => {
    if (!root || visited.has(root)) return;
    visited.add(root);
    const selector = ['button','a','input[type="button"]','input[type="submit"]','[role="button"]'].join(',');
    for (const el of Array.from(root.querySelectorAll(selector))) {
      const text = normalise(el.innerText || el.value || el.textContent || el.getAttribute('aria-label'));
      if (!patterns.some(p => p.test(text))) continue;
      const rect = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      candidates.push({
        text,
        visible: rect.width > 1 && rect.height > 1 && style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity || 1) > 0,
        disabled: Boolean(el.disabled),
        ariaDisabled: el.getAttribute('aria-disabled') || '',
        x: ox + rect.left + rect.width / 2,
        y: oy + rect.top + rect.height / 2,
        label
      });
    }
    for (const el of Array.from(root.querySelectorAll('*'))) {
      if (el.shadowRoot) walk(el.shadowRoot, ox, oy, `${label} > ${el.tagName.toLowerCase()}#shadow-root`);
      if ((el.tagName || '').toLowerCase() === 'iframe') {
        try {
          const r = el.getBoundingClientRect();
          walk(el.contentDocument, ox + r.left, oy + r.top, `${label} > iframe`);
        } catch (_) {}
      }
    }
  };
  walk(document);
  candidates.sort((a,b) => {
    const essentialA = /^Accept only essential cookies$/i.test(a.text) ? 0 : 1;
    const essentialB = /^Accept only essential cookies$/i.test(b.text) ? 0 : 1;
    if (essentialA !== essentialB) return essentialA - essentialB;
    if (a.visible !== b.visible) return a.visible ? -1 : 1;
    return 0;
  });
  return {found: candidates.length > 0, candidate: candidates[0] || null, candidates};
})()
"""

COOKIE_VERIFY_SCRIPT = r"""
(() => {
  const body = (document.body?.innerText || '').replace(/\s+/g, ' ').trim();
  return {
    hasCookieBanner: /Accept all cookies/i.test(body) && /Accept only essential cookies/i.test(body),
    bodyExcerpt: body.slice(0, 2500),
    url: location.href,
    title: document.title
  };
})()
"""


async def accept_echa_cookie_banner(
    websocket: Any,
    counter: list[int],
) -> dict[str, Any]:
    """Dismiss ECHA's cookie banner with a trusted CDP click when it is visible."""
    diagnostics: dict[str, Any] = {"seen": False, "accepted": False, "attempts": []}

    for attempt in range(1, 4):
        found_result = await raw_cdp_send(
            websocket,
            counter,
            "Runtime.evaluate",
            {"expression": COOKIE_FIND_SCRIPT, "returnByValue": True, "awaitPromise": True},
        )
        found = found_result.get("result", {}).get("value", {})
        candidate = found.get("candidate")
        diagnostics["seen"] = bool(diagnostics["seen"] or found.get("found"))
        record: dict[str, Any] = {"attempt": attempt, "find": found}

        if candidate and candidate.get("visible") and not candidate.get("disabled") and str(candidate.get("ariaDisabled", "")).lower() != "true":
            x, y = float(candidate["x"]), float(candidate["y"])
            for event, extra in [
                ("mouseMoved", {}),
                ("mousePressed", {"button": "left", "clickCount": 1}),
                ("mouseReleased", {"button": "left", "clickCount": 1}),
            ]:
                await raw_cdp_send(
                    websocket,
                    counter,
                    "Input.dispatchMouseEvent",
                    {"type": event, "x": x, "y": y, **extra},
                )
            record["trusted_click"] = {"text": candidate.get("text"), "x": x, "y": y}
            await asyncio.sleep(2.5)

        verify_result = await raw_cdp_send(
            websocket,
            counter,
            "Runtime.evaluate",
            {"expression": COOKIE_VERIFY_SCRIPT, "returnByValue": True, "awaitPromise": True},
        )
        verify = verify_result.get("result", {}).get("value", {})
        record["verify"] = verify
        diagnostics["attempts"].append(record)

        if not verify.get("hasCookieBanner"):
            diagnostics["accepted"] = diagnostics["seen"]
            diagnostics["remaining"] = False
            return diagnostics
        await asyncio.sleep(1.0)

    diagnostics["remaining"] = True
    return diagnostics


PAGE_MARKER_SCRIPT = r"""
(() => {
  const body = (document.body ? document.body.innerText : '') || '';
  const flat = body.replace(/\s+/g, ' ').trim();
  return {
    title: document.title || '',
    body: flat.slice(0, 12000),
    url: location.href,
    scrollHeight: document.documentElement.scrollHeight,
    hasIndustryHeading: /Industry classifications/i.test(flat),
    hasSubmissionText: /CLP Submission/i.test(flat),
    hasHazardCode: /(^|[^A-Za-z0-9])(H[0-9]{3}[A-Z]*|EUH[0-9]{3})([^A-Za-z0-9]|$)/i.test(flat),
    hasClassificationTable: /Hazard statement code/i.test(flat)
  };
})()
"""


async def read_page_markers(websocket: Any, counter: list[int]) -> dict[str, Any]:
    """Read the ECHA page markers that tell us whether the hazard data has rendered."""
    result = await raw_cdp_send(
        websocket,
        counter,
        "Runtime.evaluate",
        {"expression": PAGE_MARKER_SCRIPT, "returnByValue": True, "awaitPromise": True},
    )
    return result.get("result", {}).get("value", {}) or {}


def markers_are_complete(markers: dict[str, Any]) -> bool:
    """The capture is only worth taking once a real hazard statement code is on screen."""
    return bool(
        markers.get("hasIndustryHeading")
        and markers.get("hasSubmissionText")
        and markers.get("hasHazardCode")
    )


async def wait_for_echa_industry_content(
    websocket: Any,
    counter: list[int],
    *,
    timeout_seconds: float = CONTENT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Poll until the hazard data itself is in the DOM.

    The V2 notebook slept for a fixed 15 seconds here. That sleep is the reason the
    workflow intermittently produced a screenshot of an empty content column: the
    shell had painted, so every check passed, but the CLP notification request had
    not returned yet.
    """
    deadline = time.monotonic() + timeout_seconds
    latest: dict[str, Any] = {}

    while time.monotonic() < deadline:
        latest = await read_page_markers(websocket, counter)
        if markers_are_complete(latest):
            return latest
        await asyncio.sleep(1.0)

    return latest


EXPAND_PANELS_SCRIPT = r"""
(() => {
  // ECHA collapses the submission strip and some detail panels. Expanding them
  // before the capture keeps every classification inside the rendered page.
  //
  // The selector is scoped to the main content and skips anything that could
  // navigate: following a router link here would capture a different page.
  const scope = document.querySelector('main, [role="main"]') || document.body;
  const controls = Array.from(scope.querySelectorAll('[aria-expanded="false"]'))
    .filter((element) => {
      if (element.closest('nav, aside, header')) return false;
      const href = element.getAttribute('href');
      return !href || href.startsWith('#');
    })
    .slice(0, 40);

  let clicked = 0;
  for (const control of controls) {
    try {
      control.click();
      clicked += 1;
    } catch (_) {}
  }
  return {expanded: clicked, candidates: controls.length};
})()
"""


async def ensure_echa_content(
    websocket: Any,
    counter: list[int],
    target_url: str,
    *,
    max_attempts: int = MAX_PREPARE_ATTEMPTS,
) -> dict[str, Any]:
    """
    Navigate to the industry page and keep going until the hazard data is rendered.

    Two ECHA behaviours are handled explicitly:

    1. Until the legal notice is accepted, any deep link bounces back to
       https://chem.echa.europa.eu/ , so the final URL has to be checked.
    2. The first deep-link load sometimes returns the shell only. A bounded number
       of reloads recovers that without an unbounded wait.
    """
    diagnostics: dict[str, Any] = {"attempts": [], "reloads": 0}
    markers: dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        await raw_cdp_send(websocket, counter, "Page.navigate", {"url": target_url})
        await asyncio.sleep(3)

        # A pending legal notice sends the deep link back to the site root.
        bounced = await read_page_markers(websocket, counter)
        if not bounced.get("hasIndustryHeading") and "/industry/" not in str(
            bounced.get("url", "")
        ):
            consent = await accept_echa_legal_notice(websocket, counter)
            diagnostics.setdefault("recovery_consent", []).append(consent)
            await raw_cdp_send(websocket, counter, "Page.navigate", {"url": target_url})
            await asyncio.sleep(3)

        await accept_echa_cookie_banner(websocket, counter)
        markers = await wait_for_echa_industry_content(websocket, counter)

        diagnostics["attempts"].append({
            "attempt": attempt,
            "url": markers.get("url", ""),
            "has_industry_heading": bool(markers.get("hasIndustryHeading")),
            "has_submission_text": bool(markers.get("hasSubmissionText")),
            "has_hazard_code": bool(markers.get("hasHazardCode")),
            "scroll_height": markers.get("scrollHeight", 0),
        })

        if markers_are_complete(markers):
            diagnostics["markers"] = markers
            diagnostics["attempts_used"] = attempt
            return diagnostics

        diagnostics["reloads"] += 1

    diagnostics["markers"] = markers
    diagnostics["attempts_used"] = max_attempts
    return diagnostics


MIRROR_SCRIPT = r"""
(() => {
  // Snapshot the fully rendered page as a static document.
  //
  // Scripts are removed so the copy cannot re-bootstrap and blank itself, and a
  // <base> element is prepended so the original stylesheets, icons and pictograms
  // still resolve against chem.echa.europa.eu.
  const dropped = [];
  document.querySelectorAll('a, button').forEach((element) => {
    const text = (element.innerText || '').replace(/\s+/g, ' ').trim();
    if (!/^Accept (all|only essential) cookies$/i.test(text)) return;
    let node = element;
    for (let depth = 0; depth < 8 && node && node !== document.body; depth += 1) {
      const position = getComputedStyle(node).position;
      if (position === 'fixed' || position === 'sticky') {
        node.setAttribute('data-pixelrag-drop', '');
        dropped.push(node.tagName.toLowerCase());
        break;
      }
      node = node.parentElement;
    }
  });

  const clone = document.documentElement.cloneNode(true);
  clone.querySelectorAll('script, noscript, [data-pixelrag-drop]').forEach(
    (node) => node.remove()
  );

  const head = clone.querySelector('head');
  if (head) {
    const base = document.createElement('base');
    base.setAttribute('href', location.origin + '/');
    head.insertBefore(base, head.firstChild);
  }

  document.querySelectorAll('[data-pixelrag-drop]').forEach(
    (node) => node.removeAttribute('data-pixelrag-drop')
  );

  return {
    html: '<!doctype html>\n' + clone.outerHTML,
    droppedOverlays: dropped
  };
})()
"""


async def save_prepared_mirror(
    websocket: Any,
    counter: list[int],
    output_path: Path,
) -> dict[str, Any]:
    """
    Write a static copy of the prepared page to disk.

    PixelRAG always renders in a tab it creates itself, so it re-runs the whole SPA
    load and can lose the race again. This mirror is the deterministic fallback: it
    is the same rendered DOM, with the same stylesheets, but with nothing left to
    load asynchronously.
    """
    result = await raw_cdp_send(
        websocket,
        counter,
        "Runtime.evaluate",
        {"expression": MIRROR_SCRIPT, "returnByValue": True, "awaitPromise": True},
    )
    value = result.get("result", {}).get("value", {}) or {}
    html_text = value.get("html", "")

    if not html_text:
        return {"written": False, "path": "", "bytes": 0}

    output_path.write_text(html_text, encoding="utf-8")
    return {
        "written": True,
        "path": str(output_path),
        "bytes": len(html_text),
        "dropped_overlays": value.get("droppedOverlays", []),
    }


async def accept_echa_legal_notice(
    websocket: Any,
    counter: list[int],
) -> dict[str, Any]:
    """
    Accept the ECHA legal notice with a trusted CDP mouse event.

    The modal contains a scrollable legal-text panel. Some implementations
    require this panel to be scrolled before the acceptance control becomes
    actionable. The search traverses normal DOM, open shadow DOM, and
    same-origin iframes.
    """
    diagnostics: dict[str, Any] = {
        "seen": False,
        "accepted": False,
        "attempts": [],
    }

    await save_cdp_screenshot(
        websocket,
        counter,
        CONSENT_BEFORE_PATH,
    )

    for attempt in range(1, 5):
        scroll_result = await raw_cdp_send(
            websocket,
            counter,
            "Runtime.evaluate",
            {
                "expression": CONSENT_SCROLL_SCRIPT,
                "returnByValue": True,
                "awaitPromise": True,
            },
        )
        scroll_value = (
            scroll_result
            .get("result", {})
            .get("value", {})
        )

        await asyncio.sleep(1.5)

        find_result = await raw_cdp_send(
            websocket,
            counter,
            "Runtime.evaluate",
            {
                "expression": CONSENT_FIND_SCRIPT,
                "returnByValue": True,
                "awaitPromise": True,
            },
        )
        find_value = (
            find_result
            .get("result", {})
            .get("value", {})
        )

        candidate = find_value.get("candidate")
        diagnostics["seen"] = bool(
            diagnostics["seen"]
            or find_value.get("found")
            or scroll_value.get("controls")
        )

        attempt_record = {
            "attempt": attempt,
            "scroll": scroll_value,
            "find": find_value,
        }

        if candidate and candidate.get("visible"):
            x = float(candidate["x"])
            y = float(candidate["y"])

            await raw_cdp_send(
                websocket,
                counter,
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseMoved",
                    "x": x,
                    "y": y,
                },
            )
            await raw_cdp_send(
                websocket,
                counter,
                "Input.dispatchMouseEvent",
                {
                    "type": "mousePressed",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            await raw_cdp_send(
                websocket,
                counter,
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseReleased",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )

            attempt_record["trusted_click"] = {
                "x": x,
                "y": y,
                "text": candidate.get("text"),
                "label": candidate.get("label"),
            }

            await asyncio.sleep(3)

        verify_result = await raw_cdp_send(
            websocket,
            counter,
            "Runtime.evaluate",
            {
                "expression": CONSENT_VERIFY_SCRIPT,
                "returnByValue": True,
                "awaitPromise": True,
            },
        )
        verify_value = (
            verify_result
            .get("result", {})
            .get("value", {})
        )
        attempt_record["verify"] = verify_value
        diagnostics["attempts"].append(attempt_record)

        if diagnostics["seen"] and not verify_value.get("hasLegalNotice"):
            diagnostics["accepted"] = True
            break

        await asyncio.sleep(1.5)

    await save_cdp_screenshot(
        websocket,
        counter,
        CONSENT_AFTER_PATH,
    )

    return diagnostics


async def prepare_echa_cdp_session(
    cdp_url: str,
    target_url: str,
) -> dict[str, Any]:
    """
    Prepare the persistent ECHA browser session.

    The navigation/legal-notice sequence follows the working V2 notebook.
    The only additions are cookie dismissal, explicit content polling, and a
    diagnostic screenshot immediately before PixelRAG is invoked.
    """
    page_ws_url = await wait_for_cdp_page(cdp_url)

    diagnostics: dict[str, Any] = {
        "cdp_url": cdp_url,
        "target_url": target_url,
        "legal_notice_seen": False,
        "legal_notice_accepted": False,
        "cookie_banner_seen": False,
        "cookie_banner_accepted": False,
        "consent_result": {},
        "cookie_result": {},
        "page_title": "",
        "body_excerpt": "",
        "consent_before_screenshot": str(CONSENT_BEFORE_PATH),
        "consent_after_screenshot": str(CONSENT_AFTER_PATH),
        "pre_pixelrag_screenshot": str(PRE_PIXELRAG_PATH),
    }

    async with websockets.connect(
        page_ws_url,
        open_timeout=15,
        max_size=50 * 1024 * 1024,
    ) as websocket:
        counter = [0]

        await raw_cdp_send(websocket, counter, "Page.enable")
        await raw_cdp_send(websocket, counter, "Runtime.enable")
        await raw_cdp_send(websocket, counter, "Network.enable")
        await raw_cdp_send(
            websocket,
            counter,
            "Emulation.setDeviceMetricsOverride",
            {"width": 1600, "height": 1100, "deviceScaleFactor": 1, "mobile": False},
        )

        # Exact V2 bootstrap route.
        await raw_cdp_send(
            websocket,
            counter,
            "Page.navigate",
            {"url": "https://chem.echa.europa.eu/"},
        )
        await asyncio.sleep(8)

        # Remove the fixed cookie overlay first when present. This preference is
        # stored in the same Chrome profile that PixelRAG's attached tab inherits.
        cookie_diagnostics = await accept_echa_cookie_banner(websocket, counter)
        diagnostics["cookie_result"] = cookie_diagnostics
        diagnostics["cookie_banner_seen"] = bool(cookie_diagnostics.get("seen"))
        diagnostics["cookie_banner_accepted"] = bool(cookie_diagnostics.get("accepted"))

        # Exact V2 legal-notice handler.
        consent_diagnostics = await accept_echa_legal_notice(websocket, counter)
        diagnostics["consent_result"] = consent_diagnostics
        diagnostics["legal_notice_seen"] = bool(consent_diagnostics.get("seen"))
        diagnostics["legal_notice_accepted"] = bool(consent_diagnostics.get("accepted"))

        if diagnostics["legal_notice_seen"] and not diagnostics["legal_notice_accepted"]:
            raise RuntimeError(
                "The ECHA legal notice was visible, but the trusted CDP click did not dismiss it. "
                f"Inspect {CONSENT_BEFORE_PATH} and {CONSENT_AFTER_PATH}."
            )

        # Open the deep link and keep going until the hazard data is actually there.
        content_result = await ensure_echa_content(websocket, counter, target_url)
        diagnostics["content_result"] = content_result
        page_value = content_result.get("markers", {})

        # The cookie banner can reappear after deep-link navigation.
        deep_cookie = await accept_echa_cookie_banner(websocket, counter)
        diagnostics["deep_link_cookie_result"] = deep_cookie
        diagnostics["cookie_banner_seen"] = bool(
            diagnostics["cookie_banner_seen"] or deep_cookie.get("seen")
        )
        diagnostics["cookie_banner_accepted"] = bool(
            diagnostics["cookie_banner_accepted"] or deep_cookie.get("accepted")
        )

        # Open any collapsed panel, then warm the lazy-rendered page as V2 did.
        expand_result = await raw_cdp_send(
            websocket,
            counter,
            "Runtime.evaluate",
            {"expression": EXPAND_PANELS_SCRIPT, "returnByValue": True, "awaitPromise": True},
        )
        diagnostics["expanded_panels"] = expand_result.get("result", {}).get("value", {})

        for fraction in (0.2, 0.45, 0.7, 1.0, 0.0):
            await raw_cdp_send(
                websocket,
                counter,
                "Runtime.evaluate",
                {"expression": f"window.scrollTo(0, document.documentElement.scrollHeight * {fraction});"},
            )
            await asyncio.sleep(1.3)

        # Re-read state after the scroll and save a diagnostic screenshot of the page
        # BEFORE PixelRAG creates its attached capture tab.
        page_value = await read_page_markers(websocket, counter)
        await save_cdp_screenshot(websocket, counter, PRE_PIXELRAG_PATH)

        # Static snapshot used as the deterministic capture fallback.
        diagnostics["prepared_mirror"] = await save_prepared_mirror(
            websocket, counter, PREPARED_MIRROR_PATH
        )

        cookie_verify = await raw_cdp_send(
            websocket,
            counter,
            "Runtime.evaluate",
            {"expression": COOKIE_VERIFY_SCRIPT, "returnByValue": True, "awaitPromise": True},
        )
        cookie_state = cookie_verify.get("result", {}).get("value", {})

        diagnostics["page_title"] = page_value.get("title", "")
        diagnostics["body_excerpt"] = page_value.get("body", "")
        diagnostics["final_url"] = page_value.get("url", "")
        diagnostics["page_scroll_height"] = page_value.get("scrollHeight", 0)
        diagnostics["has_industry_heading"] = bool(page_value.get("hasIndustryHeading"))
        diagnostics["has_submission_text"] = bool(page_value.get("hasSubmissionText"))
        diagnostics["has_hazard_code"] = bool(page_value.get("hasHazardCode"))
        diagnostics["has_classification_table"] = bool(page_value.get("hasClassificationTable"))
        diagnostics["cookie_banner_remaining"] = bool(cookie_state.get("hasCookieBanner"))

        if not markers_are_complete(page_value):
            raise RuntimeError(
                "ECHA rendered the page shell but no hazard statement code appeared "
                f"within {CONTENT_TIMEOUT_SECONDS:.0f}s across "
                f"{MAX_PREPARE_ATTEMPTS} attempts. "
                f"heading={diagnostics['has_industry_heading']}, "
                f"submission={diagnostics['has_submission_text']}, "
                f"hazard_code={diagnostics['has_hazard_code']}. "
                f"Inspect {PRE_PIXELRAG_PATH}."
            )

    return diagnostics


def run_async_in_worker(
    async_function: Any,
    *args: Any,
    timeout: float = 300,
) -> Any:
    """Run asyncio code outside Kaggle's already-running kernel loop."""
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="pixelrag-async",
    ) as executor:
        future = executor.submit(
            lambda: asyncio.run(async_function(*args))
        )
        return future.result(timeout=timeout)


def run_sync_in_worker(
    function: Any,
    *args: Any,
    timeout: float = 300,
    **kwargs: Any,
) -> Any:
    """Run PixelRAG render_url(), which internally calls asyncio.run()."""
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="pixelrag-render",
    ) as executor:
        future = executor.submit(function, *args, **kwargs)
        return future.result(timeout=timeout)


def lookup_identity(smiles: str) -> dict[str, Any]:
    properties = "Title,IUPACName,CanonicalSMILES,InChIKey,MolecularFormula,MolecularWeight"
    try:
        r = http.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles, safe='')}/property/{properties}/JSON",
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        record = r.json()["PropertyTable"]["Properties"][0]
        cid = int(record["CID"])

        r = http.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON",
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        synonyms = r.json().get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
        cas_candidates = list(dict.fromkeys(str(x).strip() for x in synonyms if valid_cas(str(x).strip())))

        return {
            "status": "resolved",
            "cid": cid,
            "name": str(record.get("Title") or record.get("IUPACName") or "").strip(),
            "iupac_name": str(record.get("IUPACName") or "").strip(),
            "canonical_smiles": str(record.get("CanonicalSMILES") or "").strip(),
            "inchikey": str(record.get("InChIKey") or "").strip(),
            "molecular_formula": str(record.get("MolecularFormula") or "").strip(),
            "molecular_weight": record.get("MolecularWeight", ""),
            "cas": cas_candidates[0] if cas_candidates else "",
            "cas_candidates": cas_candidates,
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        }
    except Exception as error:
        return {"status": "unresolved", "error": f"{type(error).__name__}: {error}"}


TUTORIAL_ECHA_FALLBACKS = {
    "5786-21-0": {
        "matched_name": "Clozapine",
        "matched_cas": "5786-21-0",
        "matched_ec_number": "227-313-7",
        "substance_url": "https://chem.echa.europa.eu/100.024.831/overview",
        "industry_url": "https://chem.echa.europa.eu/100.024.831/industry/58986?searchText=5786-21-0",
    }
}


def lookup_echa(identity: dict[str, Any]) -> dict[str, Any]:
    prompt = {
        "task": "Find the exact official ECHA CHEM record for this substance.",
        "name": identity.get("name", ""),
        "cas": identity.get("cas", ""),
        "cas_candidates": identity.get("cas_candidates", []),
        "inchikey": identity.get("inchikey", ""),
        "rules": ["DO NOT use legacy 'echa.europa.eu/substance-information/' URLs.","Target URL MUST start with 'https://chem.echa.europa.eu/'","Prefer an exact CAS match", "Do not invent a URL"],
    }

    try:
        result, urls = structured_response(
            instructions="Find the exact ECHA CHEM record. Return not_found if it cannot be verified.",
            content=[{"type": "input_text", "text": json.dumps(prompt, ensure_ascii=False)}],
            schema_name="echa_url_resolution",
            schema=ECHA_URL_SCHEMA,
            use_web_search=True,
        )
        resolved = []
        for u in urls:
            try:
                resolved.append(
                    http.head(u, allow_redirects=True, timeout=15).url
                    if "vertexaisearch" in u else u
                )
            except Exception:
                resolved.append(u)
        urls = resolved

        valid_chem_urls = [u for u in urls if "chem.echa.europa.eu" in u]
        result["source_urls"] = valid_chem_urls
        result["raw_source_url_count"] = len(urls)
        if not result.get("industry_url"):
            result["industry_url"] = next(
                (u for u in valid_chem_urls if "/industry/" in u), ""
            )
        url = result.get("industry_url") or result.get("substance_url") or ""

        if "substance-information" in url:
            result["status"] = "not_found"
            result["notes"] = "Legacy ECHA URL detected and rejected."
        elif result.get("status") == "found" and is_echa_url(url):
            return result
    except Exception as error:
        result = {"status": "not_found", "notes": f"{type(error).__name__}: {error}"}

    for cas in [identity.get("cas", ""), *identity.get("cas_candidates", [])]:
        if cas in TUTORIAL_ECHA_FALLBACKS:
            fallback = copy.deepcopy(TUTORIAL_ECHA_FALLBACKS[cas])
            return {
                "status": "found",
                **fallback,
                "source_urls": [fallback["industry_url"]],
                "notes": "Verified tutorial fallback selected from the PubChem-resolved CAS.",
            }

    return result


def _browser_session_worker(
    target_url: str,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "target_url": target_url,
        "cdp_url": "",
        "chrome_path": "",
        "chrome_pid": None,
        "chrome_stderr_tail": "",
        "execution_mode": "PixelRAG patched Chrome + raw CDP",
    }

    chrome_process = None
    stderr_handle = None

    try:
        port = find_free_port(CDP_PORT)
        cdp_url = f"http://127.0.0.1:{port}"
        chrome_path = find_chrome(auto_install=False)

        profile_dir = Path(
            tempfile.mkdtemp(prefix="pixelrag_echa_profile_")
        )
        diagnostics["profile_dir"] = str(profile_dir)
        CHROME_STDERR_PATH.write_text("", encoding="utf-8")
        stderr_handle = CHROME_STDERR_PATH.open(
            "a",
            encoding="utf-8",
        )

        launch_command = [
            chrome_path,
            f"--remote-debugging-port={port}",
            "--headless=new",
            f"--user-data-dir={profile_dir}",
            *BROWSER_ARGS,
            "--window-size=1600,1100",
            "about:blank",
        ]

        chrome_process = subprocess.Popen(
            launch_command,
            stdout=subprocess.DEVNULL,
            stderr=stderr_handle,
            env=os.environ.copy(),
        )

        diagnostics["chrome_pid"] = chrome_process.pid
        diagnostics["chrome_path"] = chrome_path
        diagnostics["cdp_url"] = cdp_url
        diagnostics["launch_command"] = launch_command

        for _ in range(60):
            if chrome_process.poll() is not None:
                stderr_handle.flush()
                stderr_tail = CHROME_STDERR_PATH.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-5000:]
                raise RuntimeError(
                    "PixelRAG Chrome exited early with code "
                    f"{chrome_process.returncode}. "
                    f"Stderr: {stderr_tail}"
                )

            try:
                cdp_json(f"{cdp_url}/json/version")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError(
                "PixelRAG Chrome started but its CDP endpoint "
                "did not become ready."
            )

        # Budget covers the bootstrap, both consent handlers, up to
        # MAX_PREPARE_ATTEMPTS content attempts, the scroll warm-up and the mirror.
        session = run_async_in_worker(
            prepare_echa_cdp_session,
            cdp_url,
            target_url,
            timeout=480,
        )

        diagnostics.update(session)

        try:
            stderr_handle.flush()
            diagnostics["chrome_stderr_tail"] = (
                CHROME_STDERR_PATH.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-3000:]
            )
        except Exception:
            diagnostics["chrome_stderr_tail"] = ""

        CDP_SESSION_PATH.write_text(
            json.dumps(
                diagnostics,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return diagnostics

    except Exception:
        if chrome_process is not None and chrome_process.poll() is None:
            try:
                chrome_process.terminate()
                chrome_process.wait(timeout=5)
            except Exception:
                try:
                    chrome_process.kill()
                except Exception:
                    pass
        raise

    finally:
        if stderr_handle is not None:
            try:
                stderr_handle.close()
            except Exception:
                pass


def browser_session_agent(state: AgentState) -> AgentState:
    resolution = state["echa_resolution"]
    target_url = (
        resolution.get("industry_url")
        or resolution.get("substance_url")
    )

    try:
        diagnostics = run_sync_in_worker(
            _browser_session_worker,
            target_url,
            timeout=540,
        )

        return {
            "browser": diagnostics,
            "browser_status": "ready",
            "audit_log": audit_event(
                "Browser Session Agent",
                "Prepare an ECHA session with PixelRAG Chrome and raw CDP",
                "ready",
                (
                    f"title={diagnostics.get('page_title', '')!r}; "
                    f"consent={diagnostics.get('legal_notice_accepted')}; "
                    f"pid={diagnostics.get('chrome_pid')}"
                ),
            ),
        }

    except Exception as error:
        # Printed here because a caller that only forwards `browser_status` would
        # otherwise report a bare failure with no reason attached.
        print(f"Browser Session Agent failed: {type(error).__name__}: {error}")
        print(f"  Chrome stderr: {CHROME_STDERR_PATH}")
        print(f"  Last prepared screenshot: {PRE_PIXELRAG_PATH}")

        return {
            "browser": {
                "target_url": target_url,
                "error": f"{type(error).__name__}: {error}",
                "stderr_path": str(CHROME_STDERR_PATH),
                "session_path": str(CDP_SESSION_PATH),
                "execution_mode": "PixelRAG patched Chrome + raw CDP",
            },
            "browser_status": "failed",
            "audit_log": audit_event(
                "Browser Session Agent",
                "Prepare an ECHA session with PixelRAG Chrome and raw CDP",
                "failed",
                f"{type(error).__name__}: {error}",
            ),
        }


def route_after_browser(
    state: AgentState,
) -> Literal["ready", "failed"]:
    return "ready" if state.get("browser_status") == "ready" else "failed"


def tile_content_score(path: Path) -> float:
    """
    Fraction of rows in a tile's content column that carry ink.

    The failure this guards against is specific and visible: PixelRAG returns a tile
    showing the ECHA header, the side navigation and the "Industry classifications"
    heading, with an empty content column underneath. Measured on the real tiles,
    a populated capture scores about 0.33 and an empty one about 0.03, so the
    0.10 threshold sits well clear of both.
    """
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return 1.0  # Never block the capture because the check itself is unavailable.

    try:
        with Image.open(path) as image:
            grey = np.asarray(image.convert("L"))
    except Exception:
        return 0.0

    height, width = grey.shape
    left = min(340, width // 4)          # skip the side navigation
    right = max(left + 1, width - 24)    # skip the scrollbar
    top = min(230, max(0, height - 1))   # skip the header and page heading

    content = grey[top:height, left:right]
    if content.size == 0:
        return 0.0

    dark = content < 160
    return float((dark.mean(axis=1) > 0.004).mean())


def collect_tiles(tile_dirs: list[Any], diagnostics: dict[str, Any]) -> list[Path]:
    tile_paths: list[Path] = []
    for tile_dir in tile_dirs:
        tile_dir = Path(tile_dir)
        diagnostics.setdefault("tile_directories", []).append(str(tile_dir))
        for pattern in ("tile_*.jpg", "tile_*.jpeg", "tile_*.png"):
            tile_paths.extend(sorted(tile_dir.glob(pattern)))
    return [path for path in tile_paths if path.exists() and path.stat().st_size > 3_000]


def render_tiles(source_url: str, output_dir: Path, cdp_url: str) -> list[Path]:
    """Run PixelRAG exactly as the working V2 notebook did, in the prepared browser."""
    return run_sync_in_worker(
        render_url,
        source_url,
        output_dir=output_dir,
        backend="cdp",
        viewport_width=1600,
        tile_height=1800,
        quality=92,
        workers=1,
        wait_network_idle=True,
        cdp_url=cdp_url,
        # Per render, so a hung capture cannot consume the whole retry chain.
        timeout=180,
    )


def pixelrag_capture_agent(state: AgentState) -> AgentState:
    attempt = state.get("capture_attempts", 0) + 1
    browser = state["browser"]
    resolution = state["echa_resolution"]
    target_url = resolution.get("industry_url") or resolution.get("substance_url")

    attempt_dir = CAPTURE_ROOT / f"attempt_{attempt}"
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)

    diagnostics: dict[str, Any] = {
        "attempt": attempt,
        "target_url": target_url,
        "cdp_url": browser.get("cdp_url", ""),
        "tile_directories": [],
        "tile_paths": [],
        "renders": [],
        "capture_method": "",
    }

    mirror = browser.get("prepared_mirror", {}) or {}
    mirror_path = mirror.get("path", "") if mirror.get("written") else ""

    # Live renders first, then the static mirror of the prepared DOM. Each live
    # render is an independent draw against the SPA's load race, so a rejected
    # capture is retried rather than handed to the vision model.
    sources: list[tuple[str, str]] = [
        (f"live_{index}", target_url) for index in range(1, MAX_LIVE_RENDER_ATTEMPTS + 1)
    ]
    if mirror_path:
        sources.append(("prepared_mirror", f"file://{mirror_path}"))

    best_tiles: list[Path] = []
    best_score = -1.0
    best_label = ""

    try:
        for index, (label, source_url) in enumerate(sources, start=1):
            render_dir = attempt_dir / f"render_{index}_{label}"
            render_dir.mkdir(parents=True, exist_ok=True)

            record: dict[str, Any] = {"label": label, "source_url": source_url}

            try:
                tile_dirs = render_tiles(source_url, render_dir, browser["cdp_url"])
                tiles = collect_tiles(tile_dirs, diagnostics)
            except Exception as error:
                # One failed render must not cost us the remaining fallbacks.
                record.update({
                    "tile_count": 0,
                    "content_score": 0.0,
                    "accepted": False,
                    "error": f"{type(error).__name__}: {error}",
                })
                diagnostics["renders"].append(record)
                continue

            scores = [tile_content_score(path) for path in tiles]
            score = max(scores) if scores else 0.0

            record.update({
                "tile_count": len(tiles),
                "content_score": round(score, 4),
                "accepted": score >= TILE_CONTENT_ROW_THRESHOLD,
            })
            diagnostics["renders"].append(record)

            if score > best_score:
                best_score, best_tiles, best_label = score, tiles, label

            if score >= TILE_CONTENT_ROW_THRESHOLD:
                break

        best_score = max(best_score, 0.0)

        # Prefer tiles that actually carry content, then keep reading order.
        ranked = sorted(
            best_tiles,
            key=lambda path: (-tile_content_score(path), path.name),
        )[:MAX_VISION_TILES]
        usable_tiles = sorted(ranked, key=lambda path: path.name)

        diagnostics["tile_paths"] = [str(path) for path in usable_tiles]
        diagnostics["tile_count"] = len(usable_tiles)
        diagnostics["content_score"] = round(best_score, 4)
        diagnostics["capture_method"] = (
            f"PixelRAG render_url with attached CDP browser ({best_label})"
            if best_label
            else "PixelRAG render_url produced no tiles"
        )

        has_content = best_score >= TILE_CONTENT_ROW_THRESHOLD
        status = "captured" if usable_tiles and has_content else "failed"
        if not usable_tiles:
            render_errors = [
                entry["error"] for entry in diagnostics["renders"] if entry.get("error")
            ]
            diagnostics["error"] = (
                "; ".join(render_errors)
                if render_errors
                else "PixelRAG returned no usable tiles."
            )
        elif not has_content:
            diagnostics["error"] = (
                "Every render returned a page with an empty content column "
                f"(best score {best_score:.3f} < {TILE_CONTENT_ROW_THRESHOLD})."
            )

        # The render log is the diagnostic that explains a bad capture, so it is
        # printed rather than left buried in the returned state.
        print(f"PixelRAG capture attempt {attempt}: {status}")
        for entry in diagnostics["renders"]:
            line = (
                f"  {entry['label']:<16} tiles={entry['tile_count']:<2} "
                f"score={entry['content_score']:.3f} accepted={entry['accepted']}"
            )
            if entry.get("error"):
                line += f" error={entry['error']}"
            print(line)
        if diagnostics.get("error"):
            print(f"  {diagnostics['error']}")

        return {
            "capture": diagnostics,
            "capture_status": status,
            "capture_attempts": attempt,
            "audit_log": audit_event(
                "PixelRAG Capture Agent",
                f"Render the ECHA page into screenshot tiles, attempt {attempt}",
                status,
                (
                    f"Captured {len(usable_tiles)} usable tile(s) via {best_label}; "
                    f"content score {best_score:.3f}."
                ),
            ),
        }

    except Exception as error:
        diagnostics["error"] = f"{type(error).__name__}: {error}"
        return {
            "capture": diagnostics,
            "capture_status": "failed",
            "capture_attempts": attempt,
            "audit_log": audit_event(
                "PixelRAG Capture Agent",
                f"Render the ECHA page, attempt {attempt}",
                "failed",
                diagnostics["error"],
            ),
        }


def route_after_capture(state: AgentState) -> Literal["captured", "failed"]:
    return "captured" if state.get("capture_status") == "captured" else "failed"


def pixelrag_self_test() -> dict[str, Any]:
    PIXELRAG_CHROME_PATH = find_chrome(auto_install=False)

    probe_port = find_free_port()
    probe_cdp_url = f"http://127.0.0.1:{probe_port}"
    probe_profile = Path(
        tempfile.mkdtemp(prefix="pixelrag_chrome_probe_")
    )
    probe_stderr_path = OUTPUT_DIR / "pixelrag_chrome_probe_stderr.log"
    probe_stderr_path.write_text("", encoding="utf-8")

    probe_process = None
    probe_stderr = None
    probe_ready = False
    probe_error = ""

    try:
        probe_stderr = probe_stderr_path.open(
            "a",
            encoding="utf-8",
        )

        probe_command = [
            PIXELRAG_CHROME_PATH,
            f"--remote-debugging-port={probe_port}",
            "--headless=new",
            f"--user-data-dir={probe_profile}",
            *BROWSER_ARGS,
            "about:blank",
        ]

        probe_process = subprocess.Popen(
            probe_command,
            stdout=subprocess.DEVNULL,
            stderr=probe_stderr,
            env=os.environ.copy(),
        )

        # PixelRAG's renderer treats Chrome as a long-lived CDP process.
        # The correct health test is therefore CDP readiness, not process exit.
        for _ in range(60):
            if probe_process.poll() is not None:
                probe_stderr.flush()
                stderr_tail = probe_stderr_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-4000:]
                probe_error = (
                    "PixelRAG Chrome exited before CDP became ready. "
                    f"returncode={probe_process.returncode}; "
                    f"stderr={stderr_tail}"
                )
                break

            try:
                version_info = cdp_json(
                    f"{probe_cdp_url}/json/version"
                )
                if version_info.get("webSocketDebuggerUrl"):
                    probe_ready = True
                    break
            except Exception:
                pass

            time.sleep(0.5)

        if not probe_ready and not probe_error:
            probe_error = (
                "PixelRAG Chrome remained running, but the CDP endpoint "
                "did not become ready within 30 seconds."
            )

    finally:
        if probe_process is not None and probe_process.poll() is None:
            probe_process.terminate()
            try:
                probe_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                probe_process.kill()
                probe_process.wait(timeout=5)

        if probe_stderr is not None:
            try:
                probe_stderr.close()
            except Exception:
                pass

        shutil.rmtree(probe_profile, ignore_errors=True)

    print("PixelRAG Chrome:", PIXELRAG_CHROME_PATH)
    print("CDP endpoint:", probe_cdp_url)
    print("CDP ready:", probe_ready)
    print("Probe stderr:", probe_stderr_path)

    assert probe_ready, probe_error
    print("PixelRAG Chrome CDP self-test: passed")
    return {"chrome_path": PIXELRAG_CHROME_PATH, "cdp_ready": probe_ready, "cdp_url": probe_cdp_url}


def cleanup_browser(browser_state: dict[str, Any]) -> None:
    """Terminate the persistent Chromium process and remove its temporary profile."""
    chrome_pid = browser_state.get("chrome_pid")
    profile_dir = browser_state.get("profile_dir")

    if chrome_pid:
        try:
            os.kill(int(chrome_pid), signal.SIGTERM)
            print("Terminated Chromium PID:", chrome_pid)
        except ProcessLookupError:
            print("Chromium was already stopped:", chrome_pid)
        except Exception as error:
            print("Browser cleanup warning:", type(error).__name__, error)
    else:
        print("No Chromium PID was recorded.")

    if profile_dir:
        shutil.rmtree(Path(profile_dir), ignore_errors=True)
        print("Removed browser profile:", profile_dir)

    print("Browser cleanup completed.")