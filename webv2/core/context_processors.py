from django.conf import settings


def app_settings(request):
    """ส่งค่าตั้งค่าระดับแอปเข้า template ทุกหน้า (เช่น URL ของ Gradio)."""
    return {"GRADIO_URL": settings.GRADIO_URL}
