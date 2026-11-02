"""
pdf_report.py  -  PHASE 10  (PDF screening report using ReportLab)
"""
import io
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _png(rgb: np.ndarray, width_cm: float):
    buf = io.BytesIO()
    PILImage.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    h, w = rgb.shape[:2]
    width = width_cm * cm
    return Image(buf, width=width, height=width * h / w)


def build_pdf(report: dict, original_rgb: np.ndarray, overlay_rgb=None) -> bytes:
    """
    report keys: ref_id, date, patient_name (optional), quality (dict), rec (dict from the engine),
                 risk_method
    Returns the PDF as bytes.
    """
    rec = report["rec"]
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#0F766E"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#0F766E"))
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))

    def bullets(items):
        return [Paragraph("&bull; " + escape(t), body) for t in items]

    story = [Paragraph("Dental Screening Report", h1), Spacer(1, 6)]

    meta = [["Reference ID", report["ref_id"]], ["Date of screening", report["date"]]]
    if report.get("patient_name"):
        meta.append(["Patient", report["patient_name"]])
    q = report["quality"]
    meta.append(["Image quality", f"OK (sharpness {q.get('blur_score')}, brightness {q.get('brightness')}, "
                                  f"{q.get('width')}x{q.get('height')} px)"])
    t = Table(meta, colWidths=[4 * cm, 12 * cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
    story += [t, Spacer(1, 10)]

    imgs = [_png(original_rgb, 6.5)]
    if overlay_rgb is not None:
        imgs.append(_png(overlay_rgb, 6.5))
    cap = ["Uploaded photo"] + (["Areas that influenced the model (Grad-CAM)"] if overlay_rgb is not None else [])
    story += [Table([imgs, [Paragraph(c, small) for c in cap]]), Spacer(1, 10)]

    story += [Paragraph("Possible condition", h2),
              Paragraph(f"<b>{escape(rec['condition_name'])}</b> - model confidence "
                        f"{rec['confidence'] * 100:.0f}%", body),
              Paragraph(escape(rec["explanation"]), body),
              Paragraph(f"Screening risk category: <b>{escape(str(rec['risk_category']).title())}</b> "
                        f"<font size=8>({escape(report['risk_method'])})</font>", body),
              Spacer(1, 6)]

    story += [Paragraph("Dentist consultation guidance", h2),
              Paragraph(f"<b>{escape(rec['urgency_title'])}</b> - {escape(rec['urgency_text'])}", body)]
    if rec["urgent_message"]:
        story.append(Paragraph("<b>" + escape(rec["urgent_message"]) + "</b>", body))

    story += [Paragraph("Possible treatment pathways (for a dentist to evaluate)", h2)] + bullets(rec["pathways"])
    story += [Paragraph("Supportive home care", h2)] + bullets(rec["home_care"])
    story += [Paragraph("Seek urgent care if you notice", h2)] + bullets(rec["warning_signs"])
    story += [Spacer(1, 12), Paragraph("Medical disclaimer", h2), Paragraph(escape(rec["disclaimer"]), small)]

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="Dental Screening Report").build(story)
    return buf.getvalue()
