"""
轻量双语(中/英)支持。

用法
----
    from .i18n import T, set_lang, get_lang, save_lang

    print(T("晶系识别与晶格参数", "Crystal system & lattice parameters"))

语言选择优先级(get_lang 首次初始化时):
    1. 运行时 set_lang("en"/"zh") 显式设置(设置菜单、config.lang);
    2. 环境变量 MHEC_LANG = en / zh / english / chinese;
    3. 用户偏好文件 ~/.mhec/lang(由 save_lang 写入,实现"永久英文");
    4. 默认中文 (zh)。

save_lang(lang) 会把选择写入 ~/.mhec/lang,下次启动自动生效。

设计要点:T(zh, en) 直接内联中英两种文案,无需外部资源文件;
en 省略时回退到 zh,保证渐进式改造过程中不会缺字符串。
"""

import os

_LANG = None  # None 表示尚未初始化

# 用户级语言偏好文件(跨会话持久化)
_PREF_PATH = os.path.join(os.path.expanduser("~"), ".mhec", "lang")


def _normalize(val: str) -> str:
    v = (val or "").strip().lower()
    if v in ("en", "english", "eng", "en_us", "en-us"):
        return "en"
    if v in ("zh", "cn", "chinese", "zh_cn", "zh-cn", "中文"):
        return "zh"
    return ""


def _read_pref() -> str:
    """读取用户偏好文件中的语言;失败或不存在返回 ''。"""
    try:
        with open(_PREF_PATH, "r", encoding="utf-8") as f:
            return _normalize(f.read())
    except Exception:
        return ""


def get_lang() -> str:
    """返回当前语言 'zh' 或 'en'。

    首次调用时按优先级初始化:环境变量 MHEC_LANG > 用户偏好文件 > 默认中文。
    """
    global _LANG
    if _LANG is None:
        _LANG = _normalize(os.environ.get("MHEC_LANG", "")) or _read_pref() or "zh"
    return _LANG


def set_lang(lang: str) -> str:
    """显式设置本会话语言;返回规范化后的语言码。无效值忽略。"""
    global _LANG
    norm = _normalize(lang)
    if norm:
        _LANG = norm
    return get_lang()


def save_lang(lang: str) -> bool:
    """把语言选择写入用户偏好文件 ~/.mhec/lang,实现跨会话永久生效。

    同时更新本会话语言。成功返回 True。
    """
    norm = _normalize(lang)
    if not norm:
        return False
    set_lang(norm)
    try:
        os.makedirs(os.path.dirname(_PREF_PATH), exist_ok=True)
        with open(_PREF_PATH, "w", encoding="utf-8") as f:
            f.write(norm + "\n")
        return True
    except Exception:
        return False


def pref_path() -> str:
    """返回用户偏好文件路径(用于提示用户)。"""
    return _PREF_PATH


def is_en() -> bool:
    return get_lang() == "en"


def T(zh: str, en: str = None) -> str:
    """按当前语言返回文案。en 省略时回退到 zh。"""
    if get_lang() == "en" and en is not None:
        return en
    return zh
