from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .forms import AnalysisRecordForm
from .models import AnalysisRecord


def home(request):
    """หน้าแรก (เข้าได้โดยไม่ต้องล็อกอิน) — แนะนำระบบ + ทางเข้าใช้งาน."""
    return render(request, "home.html")


@login_required
def dashboard(request):
    """หน้าเครื่องมือวิเคราะห์ — ฝัง Gradio (app.py) ผ่าน iframe.

    ต้องล็อกอินก่อน จึงเป็นการ "ครอบ" เครื่องมือ Gradio ด้วยระบบสิทธิ์ของ Django
    """
    return render(request, "dashboard.html")


@login_required
def history(request):
    """ประวัติผลวิเคราะห์ของผู้ใช้ + ฟอร์มบันทึกผลใหม่ (สาธิตการใช้ฐานข้อมูล)."""
    if request.method == "POST":
        form = AnalysisRecordForm(request.POST)
        if form.is_valid():
            record = form.save(commit=False)
            record.created_by = request.user
            record.save()
            messages.success(request, "บันทึกผลวิเคราะห์เรียบร้อยแล้ว")
            return redirect("history")
    else:
        form = AnalysisRecordForm()

    records = AnalysisRecord.objects.filter(created_by=request.user)
    return render(request, "history.html", {"records": records, "form": form})
