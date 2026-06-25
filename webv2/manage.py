#!/usr/bin/env python
"""Django command-line utility สำหรับงาน admin ของ web v2."""
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "ไม่พบ Django — ติดตั้งก่อนด้วย: pip install -r requirements.txt "
            "และเปิดใช้งาน virtualenv แล้วหรือยัง?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
