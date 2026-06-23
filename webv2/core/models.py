from django.conf import settings
from django.db import models


class AnalysisRecord(models.Model):
    """บันทึกผลการวิเคราะห์วิดีโอหนึ่งครั้ง — เก็บไว้ดูย้อนหลังต่อผู้ใช้

    เว็บ v2 ใช้ตารางนี้เพื่อทำสิ่งที่ Gradio เดิมทำไม่ได้: เก็บประวัติผล
    วิเคราะห์ผูกกับผู้ใช้ที่ล็อกอิน และค้นดูภายหลังผ่านหน้า "ประวัติ" หรือ
    Django admin
    """

    title = models.CharField("ชื่อรายการ", max_length=200)
    source_filename = models.CharField("ไฟล์วิดีโอ", max_length=300, blank=True)
    objects_detected = models.PositiveIntegerField("จำนวนวัตถุที่ตรวจพบ", default=0)
    summary = models.TextField("สรุปผล", blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="records",
        verbose_name="ผู้บันทึก",
    )
    created_at = models.DateTimeField("บันทึกเมื่อ", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "บันทึกผลวิเคราะห์"
        verbose_name_plural = "บันทึกผลวิเคราะห์"

    def __str__(self):
        return f"{self.title} ({self.created_at:%Y-%m-%d %H:%M})"
