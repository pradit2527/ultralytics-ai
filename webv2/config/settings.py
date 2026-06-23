"""
การตั้งค่า Django สำหรับ web v2 (AIDC Tech Video Analytics)

เว็บ v2 ใช้ Django เป็นเปลือกหลัก: ระบบ login, จัดการผู้ใช้ (admin),
เก็บประวัติผลวิเคราะห์ลงฐานข้อมูล แล้ว "ฝัง" เครื่องมือ Gradio เดิม
(app.py ในโฟลเดอร์แม่) เข้ามาในหน้า dashboard ผ่าน iframe
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── ความปลอดภัย ──────────────────────────────────────────
# โปรดตั้ง DJANGO_SECRET_KEY เป็นค่าจริงตอนใช้งานจริง
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY", "dev-insecure-key-change-me-before-production"
)
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")
CSRF_TRUSTED_ORIGINS = [
    o for o in os.environ.get("DJANGO_CSRF_TRUSTED", "").split(",") if o
]

# ── แอป ──────────────────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "core",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "core.context_processors.app_settings",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# ── ฐานข้อมูล (SQLite สำหรับเริ่มต้น เปลี่ยนเป็น Postgres ได้ภายหลัง) ──
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ── ภาษา / เขตเวลา ───────────────────────────────────────
LANGUAGE_CODE = "th"
TIME_ZONE = "Asia/Bangkok"
USE_I18N = True
USE_TZ = True

# ── ไฟล์ static ──────────────────────────────────────────
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ── การล็อกอิน ───────────────────────────────────────────
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "dashboard"
LOGOUT_REDIRECT_URL = "home"

# ── ที่อยู่เครื่องมือ Gradio (app.py) ที่จะฝังเข้า iframe ──
# รัน Gradio แยกต่างหาก แล้วชี้ค่านี้มาที่ host:port ของมัน
GRADIO_URL = os.environ.get("GRADIO_URL", "http://127.0.0.1:7860")
