"""
轻量级文本进度条 (零外部依赖, 兼容 Windows cmd / Linux 集群终端)。

用法:
    from .progress import track, ProgressBar

    # 1) 包裹可迭代对象
    for item in track(items, desc="读取应力"):
        ...

    # 2) 手动控制
    bar = ProgressBar(total=10, desc="绘图")
    for ...:
        bar.update(1, info="E_2D_xy")
    bar.close()
"""

import sys
import time

__all__ = ["ProgressBar", "track"]


def _supports_unicode(stream) -> bool:
    enc = getattr(stream, "encoding", None) or ""
    return "utf" in enc.lower()


class ProgressBar:
    """单行刷新进度条，显示百分比、计数、已用时间与预计剩余 (ETA)。"""

    def __init__(self, total, desc="", width=28, stream=None):
        self.total = max(int(total), 1)
        self.desc = desc
        self.width = width
        self.stream = stream or sys.stdout
        self.n = 0
        self.start = time.time()
        if _supports_unicode(self.stream):
            self._fill, self._empty = "█", "░"
        else:
            self._fill, self._empty = "#", "-"
        self._closed = False
        self._render()

    def update(self, step=1, info=""):
        self.n += step
        self._render(info)

    def _fmt_time(self, sec):
        sec = int(max(sec, 0))
        if sec < 60:
            return f"{sec}s"
        m, s = divmod(sec, 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"

    def _render(self, info=""):
        frac = min(self.n / self.total, 1.0)
        filled = int(round(self.width * frac))
        bar = self._fill * filled + self._empty * (self.width - filled)
        elapsed = time.time() - self.start
        eta = (elapsed / frac - elapsed) if frac > 1e-9 else 0.0
        tail = f" {info}" if info else ""
        msg = (f"\r  {self.desc} |{bar}| {frac*100:5.1f}% "
               f"({self.n}/{self.total}) "
               f"已用 {self._fmt_time(elapsed)} 剩 {self._fmt_time(eta)}{tail}")
        # 末尾补空格覆盖上一次更长的 info
        self.stream.write(msg + "    ")
        self.stream.flush()

    def close(self, info="完成"):
        if self._closed:
            return
        self.n = self.total
        self._render(info)
        self.stream.write("\n")
        self.stream.flush()
        self._closed = True


def track(iterable, desc="", total=None, stream=None):
    """包裹可迭代对象，自动显示进度条。total 未知时退化为普通迭代。"""
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    if not total:
        for x in iterable:
            yield x
        return
    bar = ProgressBar(total, desc=desc, stream=stream)
    try:
        for x in iterable:
            yield x
            bar.update()
    finally:
        bar.close()
