"""
apps/proxy/ts_proxy/ssai_preprocessor.py

SSAI (Server-Side Ad Insertion) manifest preprocessor for Dispatcharr.

Three responsibilities:
  1. Resolve HLS master playlists to a single concrete rendition URL before
     handing off to FFmpeg — eliminates rendition ambiguity and prevents FFmpeg
     from silently picking the wrong (often lowest) quality track.
  2. Detect SSAI sources (Pluto TV, Tubi, etc.) and signal the stream manager
     to enable DTS-continuity FFmpeg flags that treat ad/content boundary jumps
     as splices rather than corrupt input.
  3. Classify FFmpeg stderr lines that look like errors but are actually normal
     SSAI splice noise, so the health monitor doesn't trigger needless failovers.

Drop in as apps/proxy/ts_proxy/ssai_preprocessor.py.
Import in stream_manager.py as shown at the bottom of this file.
"""

import re
import logging
import time
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
# SSAI detection patterns
# ---------------------------------------------------------------------------

# Manifest-level strings that reliably indicate SSAI
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
]

# FFmpeg stderr patterns that are SSAI splice noise, not real errors.
# Each is a compiled regex checked against the lowercased stderr line.
_SSAI_STDERR_NOISE_PATTERNS = [
    re.compile(r"dts .{0,40} out of order"),
    re.compile(r"pts .{0,40} out of order"),
    re.compile(r"non.monoton"),                 # "non monotonous DTS"
    re.compile(r"dts .{0,20}, next: .{0,20} st:"),
    re.compile(r"application provided invalid"),
    re.compile(r"pts has no value"),
    re.compile(r"st: \d+, invalid"),
    re.compile(r"discarding"),                   # "discarding corrupt packet"
    re.compile(r"discontinuity detected"),
    re.compile(r"missing pts"),
]

# FFmpeg input-side flags to inject for SSAI sources.
# Inserted immediately before the first -i in the command list.
_SSAI_INPUT_FLAGS = [
    "-fflags",        "+genpts+discardcorrupt+igndts",
    "-avoid_negative_ts", "make_zero",
    "-ignore_unknown",
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SSAIPreprocessor:
    """
    Resolves HLS master playlists and detects SSAI sources.

    Typical usage inside stream_manager._establish_transcode_connection():

        preprocessor = SSAIPreprocessor()
        resolved_url, is_ssai, meta = preprocessor.detect_and_resolve(
            self.url, self.user_agent
        )
        if resolved_url != self.url:
            logger.info(f"SSAI: resolved master → {resolved_url}")
            self.url = resolved_url
        self.ssai_mode = is_ssai

        self.transcode_cmd = stream_profile.build_command(self.url, self.user_agent)

        if self.ssai_mode:
            self.transcode_cmd = SSAIPreprocessor.inject_ssai_flags(self.transcode_cmd)
    """

    # Rendition height preference. If you want to prefer 720p to save CPU,
    # lower this.  The selector picks the best rendition ≤ preferred_height,
    # or the lowest rendition above it if none qualify.
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
        Inspect *url* and, if it is an HLS master playlist, resolve it to the
        best single-rendition media playlist URL.

        Returns
        -------
        (resolved_url, is_ssai, metadata)

        resolved_url : str
            The concrete media playlist URL, or the original URL if resolution
            was not needed / not possible.
        is_ssai : bool
            True when SSAI markers were detected in the manifest body.
        metadata : dict
            Extra hints the caller may use:
              'is_master'     – was the original URL a master playlist?
              'ssai_mode'     – same as is_ssai (convenience copy)
              'audio_url'     – separate audio rendition URL, if any
              'original_url'  – the unmodified input URL
              'resolved_url'  – the selected rendition URL
        """
        meta: Dict[str, Any] = {
            "is_master": False,
            "ssai_mode": False,
            "audio_url": None,
            "original_url": url,
            "resolved_url": url,
        }

        try:
            headers = {"User-Agent": user_agent}
            resp = self._session.get(
                url, headers=headers, timeout=self._timeout, stream=True
            )
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").lower()
            # Read the manifest body – cap at 256 KB to avoid runaway reads
            body = resp.content[:262144].decode("utf-8", errors="replace")
            resp.close()

            if not self._is_hls_content(content_type, body):
                logger.debug(f"SSAI: {url!r} does not appear to be HLS, skipping")
                return url, False, meta

            is_ssai = self._detect_ssai_markers(body)
            meta["ssai_mode"] = is_ssai

            if "#EXT-X-STREAM-INF" not in body:
                # Already a media playlist — nothing to resolve
                logger.debug(f"SSAI: {url!r} is a media playlist (ssai={is_ssai})")
                return url, is_ssai, meta

            # ---- Master playlist ----------------------------------------
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
            meta["audio_group"] = best.audio_group

            # Look for a dedicated audio rendition (muxed into the video
            # playlist already on most sources, but some SSAI providers
            # keep a separate EXT-X-MEDIA audio group)
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
    # Static helpers (usable without an instance)
    # ------------------------------------------------------------------

    @staticmethod
    def is_ssai_stderr_noise(line: str) -> bool:
        """
        Return True when *line* (FFmpeg stderr) is SSAI splice noise rather
        than a real error.  Call this in stream_manager._log_stderr_content()
        to suppress spurious error-level log entries and prevent the health
        monitor from triggering a failover on every ad break.
        """
        ll = line.lower()
        return any(p.search(ll) for p in _SSAI_STDERR_NOISE_PATTERNS)

    @staticmethod
    def inject_ssai_flags(cmd: list) -> list:
        """
        Rewrite an FFmpeg command list to include SSAI-safe input flags.

        The injected flags are inserted immediately before the first ``-i``
        argument so they apply to the input session, not a later output::

            ffmpeg -fflags +genpts+discardcorrupt+igndts
                   -avoid_negative_ts make_zero
                   -ignore_unknown
                   -i <url> …

        If ``-i`` is not found the flags are prepended after the binary name.
        Returns the original list unchanged if it is empty.
        """
        if not cmd:
            return cmd

        result = [cmd[0]]          # ffmpeg / cvlc binary — keep as-is
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
        """Parse an HLS attribute list string into a plain dict."""
        # Strip tag prefix (everything up to and including the first ':')
        if ":" in attr_string:
            attr_string = attr_string.split(":", 1)[1]
        attrs: Dict[str, str] = {}
        # Match KEY=VALUE where VALUE is either quoted or unquoted
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
                # Skip blank lines to reach the rendition URI
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
    def _select_best_variant(
        variants: list, preferred_height: int
    ) -> HLSVariant:
        """
        Selection strategy:
          1. Among renditions with height ≤ preferred_height, take the tallest
             (highest fidelity that won't exceed the preference), breaking ties
             by highest bandwidth.
          2. If nothing is ≤ preferred_height, take the *shortest* rendition
             above it (closest to the preference), breaking ties by highest
             bandwidth.
          3. If no resolution info is present, take the highest bandwidth.
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
        """Return the URI of the DEFAULT=YES audio rendition for *audio_group*."""
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


# ---------------------------------------------------------------------------
# stream_manager.py integration guide (copy the snippets below)
# ---------------------------------------------------------------------------
#
# 1. IMPORT  (top of stream_manager.py, with other local imports)
# ----------------------------------------------------------------
#   from .ssai_preprocessor import SSAIPreprocessor
#
#
# 2. __init__  (inside StreamManager.__init__, after existing attrs)
# -----------------------------------------------------------------
#   # SSAI support
#   self.ssai_mode = False
#   self._ssai_preprocessor = SSAIPreprocessor()
#
#
# 3. _establish_transcode_connection  (before build_command call)
# --------------------------------------------------------------
#   # --- SSAI: resolve master playlist & detect ad-insertion sources ---
#   if hasattr(self, 'stream_type') and self.stream_type == StreamType.HLS:
#       resolved_url, is_ssai, ssai_meta = self._ssai_preprocessor.detect_and_resolve(
#           self.url, self.user_agent
#       )
#       if resolved_url != self.url:
#           logger.info(
#               f"SSAI: resolved master playlist to rendition URL "
#               f"for channel {self.channel_id}: {resolved_url[:80]}"
#           )
#           self.url = resolved_url
#       self.ssai_mode = is_ssai
#       if is_ssai:
#           logger.info(
#               f"SSAI source detected for channel {self.channel_id} "
#               f"— enabling DTS-continuity flags"
#           )
#   # -------------------------------------------------------------------
#
#   self.transcode_cmd = stream_profile.build_command(self.url, self.user_agent)
#
#   # Inject SSAI-safe FFmpeg flags AFTER build_command
#   if self.ssai_mode and self.stream_command and self.stream_command.lower() == 'ffmpeg':
#       self.transcode_cmd = SSAIPreprocessor.inject_ssai_flags(self.transcode_cmd)
#       logger.info(f"SSAI: injected DTS-continuity flags for channel {self.channel_id}")
#
#
# 4. _log_stderr_content  (at the very top of the method, before any routing)
# ---------------------------------------------------------------------------
#   # Suppress SSAI splice noise so the health monitor doesn't false-positive
#   if getattr(self, 'ssai_mode', False) and SSAIPreprocessor.is_ssai_stderr_noise(content):
#       logger.debug(
#           f"SSAI splice noise suppressed for channel {self.channel_id}: "
#           f"{content[:120]}"
#       )
#       return
#
# ---------------------------------------------------------------------------
