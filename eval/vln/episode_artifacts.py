"""Record a trial's navigation video and its corresponding model outputs."""

from contextlib import suppress
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

import cv2
import numpy as np
from habitat.utils.visualizations import maps


class EpisodeArtifacts:
    def __init__(self, result_path, pass_k):
        self._directory = Path(result_path) / "video" / "eval"
        self._pass_k = pass_k
        self._encoder = None
        self._video_tmp = None
        self._json_tmp = None

    def start(self, episode_id, instruction, trial_id):
        self._stem = quote(str(episode_id), safe="")
        if self._pass_k > 1:
            self._stem = f"{self._stem}_{trial_id}"
        self._directory.mkdir(parents=True, exist_ok=True)
        self._video_tmp = self._directory / f".{self._stem}.mp4"
        self._json_tmp = self._directory / f".{self._stem}.json"
        self._frame_index = -1
        self._trace = {"episode_id": episode_id, "instruction": instruction, "outputs": []}

    def write_frame(self, observations, info):
        frame = _render_frame(observations["rgb"], info["top_down_map"], self._trace["instruction"])
        if self._encoder is None:
            from imageio_ffmpeg import get_ffmpeg_exe

            height, width = frame.shape[:2]
            self._encoder = subprocess.Popen(
                [
                    get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", f"{width}x{height}", "-framerate", "1", "-i", "pipe:0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "25", "-threads", "1",
                    self._video_tmp,
                ],
                stdin=subprocess.PIPE,
            )
        self._encoder.stdin.write(memoryview(frame))
        self._frame_index += 1

    def record_output(self, model_output):
        self._trace["outputs"].append({
            "current_frame_index": self._frame_index,
            "model_output": model_output,
        })

    def finish(self, success):
        self._encoder.stdin.close()
        returncode = self._encoder.wait()
        self._encoder = None
        if returncode != 0:
            raise RuntimeError(f"FFmpeg exited with code {returncode} while encoding {self._stem}.mp4")

        self._json_tmp.write_text(json.dumps(self._trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outcome, other_outcome = ("success", "failure") if float(success) > 0 else ("failure", "success")
        destination = self._directory / outcome
        destination.mkdir(exist_ok=True)
        video_path = destination / f"{self._stem}.mp4"
        self._video_tmp.replace(video_path)
        try:
            self._json_tmp.replace(destination / f"{self._stem}.json")
        except OSError:
            video_path.unlink()
            raise

        # A retry after an interrupted publish can finish with a different outcome.
        for extension in ("mp4", "json"):
            (self._directory / other_outcome / f"{self._stem}.{extension}").unlink(missing_ok=True)
        self._video_tmp = self._json_tmp = None

    def close(self):
        if self._encoder is not None:
            self._encoder.kill()
            self._encoder.wait()
            with suppress(BrokenPipeError):
                self._encoder.stdin.close()
            self._encoder = None
        for path in (self._video_tmp, self._json_tmp):
            if path is not None:
                path.unlink(missing_ok=True)
        self._video_tmp = self._json_tmp = None


def _render_frame(rgb, map_info, instruction):
    height, width = rgb.shape[:2]
    topdown = maps.colorize_topdown_map(map_info["map"], map_info["fog_of_war_mask"], 0.75)
    maps.draw_agent(
        topdown, map_info["agent_map_coord"][0], map_info["agent_angle"][0],
        agent_radius_px=min(topdown.shape[:2]) // 24,
    )
    if topdown.shape[0] > topdown.shape[1]:
        topdown = np.rot90(topdown)

    # Match RLinf's side-by-side views and instruction panel, aligned for H.264.
    panel_height = max(64, int(height * 0.22))
    padded_height = ((height + panel_height + 15) // 16) * 16
    padded_width = ((2 * width + 15) // 16) * 16
    frame = np.full((padded_height, padded_width, 3), 255, dtype=np.uint8)
    frame[:height, :width] = rgb[:, :, :3]
    frame[:height, width:2 * width] = cv2.resize(topdown, (width, height), interpolation=cv2.INTER_LINEAR)
    _draw_instruction(frame[height:height + panel_height, :2 * width], instruction)
    return frame


def _draw_instruction(panel, instruction):
    text = " ".join(instruction.split())
    if not text:
        return
    if text.isupper():
        text = text.capitalize()

    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = max(8, panel.shape[1] // 100)
    max_width = panel.shape[1] - 2 * margin

    def text_width(line):
        return cv2.getTextSize(line, font, scale, 1)[0][0]

    for scale in (0.55, 0.48, 0.42, 0.36):
        line_height = cv2.getTextSize("Ag", font, scale, 1)[0][1] + 6
        max_lines = max(1, (panel.shape[0] - 2 * margin) // line_height)
        lines = []
        line = ""
        for word in text.split():
            candidate = f"{line} {word}" if line else word
            if line and text_width(candidate) > max_width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        if len(lines) <= max_lines and all(text_width(line) <= max_width for line in lines):
            break

    for index, line in enumerate(lines[:max_lines], start=1):
        if text_width(line) > max_width or (index == max_lines and len(lines) > max_lines):
            while line and text_width(line + "...") > max_width:
                line = line[:-1]
            line += "..."
        cv2.putText(
            panel, line, (margin, margin + index * line_height), font, scale,
            (0, 0, 0), 1, cv2.LINE_AA,
        )
