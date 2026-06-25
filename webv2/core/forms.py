from django import forms

from .models import AnalysisRecord


class AnalysisRecordForm(forms.ModelForm):
    class Meta:
        model = AnalysisRecord
        fields = ["title", "source_filename", "objects_detected", "summary"]
        widgets = {
            "title": forms.TextInput(attrs={"placeholder": "เช่น ตรวจการณ์พื้นที่ A"}),
            "source_filename": forms.TextInput(attrs={"placeholder": "เช่น patrol_01.mp4"}),
            "summary": forms.Textarea(attrs={"rows": 3, "placeholder": "สรุปสั้น ๆ"}),
        }
