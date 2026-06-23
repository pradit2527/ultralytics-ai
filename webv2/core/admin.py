from django.contrib import admin

from .models import AnalysisRecord


@admin.register(AnalysisRecord)
class AnalysisRecordAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "source_filename",
        "objects_detected",
        "created_by",
        "created_at",
    )
    list_filter = ("created_by", "created_at")
    search_fields = ("title", "source_filename", "summary")
    date_hierarchy = "created_at"
