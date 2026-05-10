"""
apps/proxy/ts_proxy/ssai_preprocessor.py

SSAI (Server-Side Ad Insertion) manifest preprocessor for Dispatcharr.

Three responsibilities:
  1. Resolve HLS master playlists to a single concrete rendition URL before
     handing off to FFmpeg — eliminates rendition ambiguity and prevents FFmpeg
     from silently picking the wrong (often lowest) quality track.
  2. Detect SSAI sources and signal the stream manager to enable DTS-continuity
     FFmpeg flags that treat ad/content boundary jumps as splices rather than
     corrupt input.  Detection works two ways:
       a. Manifest marker scan — looks for SCTE-35, EXT-X-DATERANGE, etc. in
          the playlist body (works on raw Pluto/Tubi manifests).
       b. Force-host list — when the manifest passes through a local proxy
          (e.g. TVNow at 192.168.1.254) the markers are stripped, so we force
          SSAI mode by hostname instead.
  3. Classify FFmpeg stderr lines that look like errors but are actually normal
     SSAI splice noise, so the health monitor doesn't trigger needless failovers.
"""

import re
import logging
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urljoin, urlparse

import requests

logger = logging.getLogger("ts_proxy.ssai_preprocessor")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class HLSVariant:
    """One #EXT-X-STREAM-INF entry from a master playlist."""

    def __init__(
        self,
        url: str,
        bandwidth: int = 0,
        resolution: str = "",
        codecs: str = "",
        audio_group: str = "",
        frame_rate: float = 0.0,
    ):
        self.url = url
        self.bandwidth = bandwidth
        self.resolution = resolution
        self.codecs = codecs
        self.audio_group = audio_group
        self.frame_rate = frame_rate

    @property
    def height(self) -> int:
        if self.resolution:
            parts = self.resolution.lower().split("x")
            try:
                return int(parts[1]) if len(parts) == 2 else 0
            except ValueError:
                return 0
        return 0

    def __repr__(self):
        return (
            f"<HLSVariant bw={self.bandwidth} res={self.resolution or '?'} "
            f"url={self.url[:60]}>"
        )


# ---------------------------------------------------------------------------
# Force-host list
#
# Hosts listed here are unconditionally treated as SSAI sources regardless of
# whether their manifests contain detectable markers.  Add your TVNow/proxy
# host here when its manifests strip SSAI tags before Dispatcharr sees them.
# ---------------------------------------------------------------------------

_SSAI_FORCE_HOSTS = {
    # TVNow local proxy — Pluto manifests arrive here with SCTE-35 stripped
    "192.168.1.254",
    # Pluto TV CDN hostnames — used when Dispatcharr gets a raw Pluto URL
    "silo.pluto.tv",
    "siloh-ns1.plutotv.net",
    "content.plutotv.net",
}


# ---------------------------------------------------------------------------
# Manifest-level SSAI marker strings
#
# Checked case-insensitively against the raw manifest body.  A single match
# is enough to declare the source an SSAI stream.
# ---------------------------------------------------------------------------

_SSAI_MANIFEST_MARKERS = [
    "scte35",
    "ext-x-daterange",
    "ext-x-cue",
    "ext-oatcls-scte35",
    "ext-x-asset",
    "dai.google.com",
    "/ssai/",
    "ssai.",
    "adpod",
    "ext-x-splicepoint",
    # Pluto-specific — present in raw CDN manifests
    "pluto.tv",
    "siloh-ns1.plutotv.net",
    # Tubi / Fox SSAI
    "tubi.tv/vast",
    "foxdcg.com",
]


# ---------------------------------------------------------------------------
# FFmpeg stderr noise patterns
#
# Lines matching any of these during SSAI playback are splice artifacts, not
# real errors.  They are suppressed so the health monitor never mistakes a
# normal ad-break DTS reset for a stream failure.
# ---------------------------------------------------------------------------

_SSAI_STDERR_NOISE_PATTERNS = [
    re.compile(r"dts .{0,40} out of order"),
    re.compile(r"pts .{0,40} out of order"),
    re.compile(r"non.monoton"),                       # "non monotonous DTS"
    re.compile(r"dts .{0,20}, next: .{0,20} st:"),
    re.compile(r"application provided invalid"),
    re.compile(r"pts has no value"),
    re.compile(r"st: \d+, invalid"),
    re.compile(r"packet corrupt"),                    # "Packet corrupt (stream=2…)"
    re.compile(r"discarding"),                        # "discarding corrupt packet"
    re.compile(r"discontinuity detected"),
    re.compile(r"missing pts"),
    re.compile(r"timestamp discontinuity"),           # "[aist#0:0] timestamp discontinuity"
    re.compile(r"new data stream"),                   # "[in#0/hls] New data stream with index"
    re.compile(r"new offset="),                       # "new offset= 731048000"
    re.compile(r"dropping it"),                       # "dropping it." from corrupt packet log
]


# ---------------------------------------------------------------------------
# FFmpeg input-side flags injected for all SSAI sources
# ---------------------------------------------------------------------------

_SSAI_INPUT_FLAGS = [
    "-fflags",            "+genpts+discardcorrupt+igndts",
    "-avoid_negative_ts", "make_zero",
    "-ignore_unknown",
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SSAIPreprocessor:
    """
    Resolves HLS master playlists and detects SSAI sources.

    Call detect_and_resolve() once per stream start (before build_command).
    Use inject_ssai_flags() to rewrite the FFmpeg command when is_ssai=True.
    Use is_ssai_stderr_noise() in _log_stderr_content() to suppress splice noise.
    """

    DEFAULT_PREFERRED_HEIGHT = 1080

    def __init__(self, session: Optional[requests.Session] = None, timeout: float = 8.0):
        self._session = session or requests.Session()
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_resolve(
        self,
        url: str,
        user_agent: str,
        preferred_height: int = DEFAULT_PREFERRED_HEIGHT,
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Inspect *url* and resolve HLS master playlists to a single rendition.

        Force-host check runs first so that even when a local proxy strips SSAI
        markers from the manifest body, the DTS-continuity flags still get armed.

        Returns
        -------
        (resolved_url, is_ssai, metadata)
        """
        meta: Dict[str, Any] = {
            "is_master": False,
            "ssai_mode": False,
            "audio_url": None,
            "original_url": url,
            "resolved_url": url,
        }

        # ------------------------------------------------------------------
        # Step 1 — force-host check
        # If the URL's hostname is in _SSAI_FORCE_HOSTS we know it is an SSAI
        # source regardless of manifest content.  We still fetch the manifest
        # to resolve master → rendition, but skip the marker scan.
        # ------------------------------------------------------------------
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        force_ssai = hostname in _SSAI_FORCE_HOSTS

        if force_ssai:
            logger.info(
                f"SSAI: force-enabled for known SSAI host {hostname!r} — "
                f"DTS-continuity flags will be injected"
            )
            meta["ssai_mode"] = True
            resolved_url = self._resolve_master_only(url, user_agent, preferred_height, meta)
            return resolved_url, True, meta

        # ------------------------------------------------------------------
        # Step 2 — fetch manifest and run marker scan + rendition resolution
        # ------------------------------------------------------------------
        try:
            headers = {"User-Agent": user_agent}
            resp = self._session.get(
                url, headers=headers, timeout=self._timeout, stream=True
            )
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").lower()
            body = resp.content[:262144].decode("utf-8", errors="replace")
            resp.close()

            if not self._is_hls_content(content_type, body):
                logger.debug(f"SSAI: {url!r} is not HLS, skipping")
                return url, False, meta

            is_ssai = self._detect_ssai_markers(body)
            meta["ssai_mode"] = is_ssai

            if "#EXT-X-STREAM-INF" not in body:
                # Already a media playlist — nothing to resolve
                logger.debug(f"SSAI: {url!r} is a media playlist (ssai={is_ssai})")
                return url, is_ssai, meta

            # Master playlist — resolve to best rendition
            meta["is_master"] = True
            variants = self._parse_master_playlist(body, url)

            if not variants:
                logger.warning(f"SSAI: master playlist has no renditions — {url!r}")
                return url, is_ssai, meta

            best = self._select_best_variant(variants, preferred_height)
            logger.info(
                f"SSAI: master → rendition  bw={best.bandwidth}  "
                f"res={best.resolution or '?'}  {best.url[:80]}"
            )
            meta["resolved_url"] = best.url

            if best.audio_group:
                audio_url = self._find_default_audio_url(body, best.audio_group, url)
                if audio_url:
                    meta["audio_url"] = audio_url
                    logger.info(f"SSAI: separate audio rendition → {audio_url[:80]}")

            return best.url, is_ssai, meta

        except requests.Timeout:
            logger.warning(f"SSAI: timeout fetching manifest {url!r} — using original")
            return url, False, meta
        except Exception as exc:
            logger.warning(f"SSAI: error processing manifest {url!r}: {exc}")
            return url, False, meta

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_ssai_stderr_noise(line: str) -> bool:
        """
        Return True when *line* is SSAI splice noise rather than a real error.
        Call this in stream_manager._log_stderr_content() before routing to
        the error logger or health monitor.
        """
        ll = line.lower()
        return any(p.search(ll) for p in _SSAI_STDERR_NOISE_PATTERNS)

    @staticmethod
    def inject_ssai_flags(cmd: list) -> list:
        """
        Rewrite an FFmpeg command list to include SSAI-safe input flags,
        inserted immediately before the first -i argument.

        ffmpeg [existing flags…] -fflags +genpts+discardcorrupt+igndts
               -avoid_negative_ts make_zero -ignore_unknown -i <url> …
        """
        if not cmd:
            return cmd

        result = [cmd[0]]   # binary name unchanged
        rest = cmd[1:]

        i_idx = next((i for i, a in enumerate(rest) if a == "-i"), None)

        if i_idx is not None:
            result.extend(rest[:i_idx])
            result.extend(_SSAI_INPUT_FLAGS)
            result.extend(rest[i_idx:])
        else:
            result.extend(_SSAI_INPUT_FLAGS)
            result.extend(rest)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_master_only(
        self,
        url: str,
        user_agent: str,
        preferred_height: int,
        meta: Dict[str, Any],
    ) -> str:
        """
        Fetch the manifest for a force-SSAI host and resolve master → rendition
        without running the marker scan (we already know it's SSAI).
        Returns the resolved URL, or the original URL on any error.
        """
        try:
            headers = {"User-Agent": user_agent}
            resp = self._session.get(
                url, headers=headers, timeout=self._timeout, stream=True
            )
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "").lower()
            body = resp.content[:262144].decode("utf-8", errors="replace")
            resp.close()

            if not self._is_hls_content(content_type, body):
                return url

            if "#EXT-X-STREAM-INF" not in body:
                return url

            meta["is_master"] = True
            variants = self._parse_master_playlist(body, url)
            if not variants:
                return url

            best = self._select_best_variant(variants, preferred_height)
            logger.info(
                f"SSAI: master → rendition  bw={best.bandwidth}  "
                f"res={best.resolution or '?'}  {best.url[:80]}"
            )
            meta["resolved_url"] = best.url

            if best.audio_group:
                audio_url = self._find_default_audio_url(body, best.audio_group, url)
                if audio_url:
                    meta["audio_url"] = audio_url

            return best.url

        except Exception as exc:
            logger.warning(
                f"SSAI: manifest fetch failed for force-host {url!r}: {exc} — "
                f"continuing with original URL"
            )
            return url

    @staticmethod
    def _is_hls_content(content_type: str, body: str) -> bool:
        hls_types = {
            "application/vnd.apple.mpegurl",
            "application/x-mpegurl",
            "audio/mpegurl",
        }
        if any(t in content_type for t in hls_types):
            return True
        return body.lstrip().startswith("#EXTM3U")

    @staticmethod
    def _detect_ssai_markers(body: str) -> bool:
        bl = body.lower()
        for marker in _SSAI_MANIFEST_MARKERS:
            if marker in bl:
                logger.debug(f"SSAI marker detected: {marker!r}")
                return True
        return False

    @staticmethod
    def _parse_attributes(attr_string: str) -> Dict[str, str]:
        if ":" in attr_string:
            attr_string = attr_string.split(":", 1)[1]
        attrs: Dict[str, str] = {}
        for m in re.finditer(
            r'([A-Z0-9_-]+)=("(?:[^"\\]|\\.)*"|[^,]+)', attr_string
        ):
            attrs[m.group(1)] = m.group(2).strip('"')
        return attrs

    def _parse_master_playlist(self, body: str, base_url: str):
        variants = []
        lines = body.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#EXT-X-STREAM-INF"):
                attrs = self._parse_attributes(line)
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines):
                    uri = lines[i].strip()
                    if uri and not uri.startswith("#"):
                        try:
                            bw = int(attrs.get("BANDWIDTH", 0))
                        except ValueError:
                            bw = 0
                        try:
                            fr = float(attrs.get("FRAME-RATE", 0))
                        except ValueError:
                            fr = 0.0
                        variants.append(
                            HLSVariant(
                                url=urljoin(base_url, uri),
                                bandwidth=bw,
                                resolution=attrs.get("RESOLUTION", ""),
                                codecs=attrs.get("CODECS", ""),
                                audio_group=attrs.get("AUDIO", "").strip('"'),
                                frame_rate=fr,
                            )
                        )
            i += 1
        return variants

    @staticmethod
    def _select_best_variant(variants: list, preferred_height: int) -> HLSVariant:
        """
        Pick best rendition:
          1. Tallest variant with height <= preferred_height (highest quality
             that doesn't exceed the preference), tiebreak by bandwidth.
          2. If nothing qualifies, the shortest variant above preferred_height.
          3. If no resolution data at all, highest bandwidth.
        """
        with_res = [v for v in variants if v.height > 0]
        no_res   = [v for v in variants if v.height == 0]

        under = [v for v in with_res if v.height <= preferred_height]
        if under:
            return max(under, key=lambda v: (v.height, v.bandwidth))

        over = [v for v in with_res if v.height > preferred_height]
        if over:
            return min(over, key=lambda v: (v.height, -v.bandwidth))

        if no_res:
            return max(no_res, key=lambda v: v.bandwidth)

        return variants[0]

    def _find_default_audio_url(
        self, body: str, audio_group: str, base_url: str
    ) -> Optional[str]:
        for line in body.splitlines():
            if "#EXT-X-MEDIA" not in line:
                continue
            attrs = self._parse_attributes(line)
            if (
                attrs.get("TYPE") == "AUDIO"
                and attrs.get("GROUP-ID", "").strip('"') == audio_group
                and attrs.get("DEFAULT", "NO").upper() == "YES"
            ):
                uri = attrs.get("URI", "").strip('"')
                if uri:
                    return urljoin(base_url, uri)
        return None
