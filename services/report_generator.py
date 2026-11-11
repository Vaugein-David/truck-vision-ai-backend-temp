"""Génération des rapports PDF (fpdf2)."""
from __future__ import annotations

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from services import image_processor as ip

NAVY = (15, 39, 71)
GREY = (100, 110, 125)
LEVEL = {"ok": (22, 140, 70), "warn": (217, 119, 6), "bad": (200, 38, 38), "neutral": (100, 110, 125)}
REPL = {"’": "'", "‘": "'", "“": '"', "”": '"', "…": "...", "—": "-", "–": "-", "œ": "oe", "Œ": "OE", " ": " ", " ": " "}


def T(value) -> str:
    s = "" if value is None else str(value)
    for a, b in REPL.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def __init__(self, demo: bool = False) -> None:
        super().__init__(format="A4")
        self.demo = demo
        self.set_auto_page_break(True, 18)
        self.set_margins(15, 15, 15)
        self.add_page()

    def header(self) -> None:
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 24, "F")
        self.set_fill_color(255, 255, 255)
        self.rect(15, 7, 10, 10, "F")
        self.set_fill_color(*NAVY)
        self.rect(17, 11, 4, 4, "F")
        self.rect(21.5, 12.5, 2, 2.5, "F")
        self.set_xy(29, 7)
        self.set_font("helvetica", "B", 15)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "TRUCK VISION AI")
        if self.demo:
            self.set_xy(120, 7)
            self.set_font("helvetica", "B", 10)
            self.set_text_color(255, 190, 90)
            self.cell(75, 10, T("MODE DEMO - DONNÉES FICTIVES"), align="R")
        self.set_text_color(0, 0, 0)
        self.set_y(30)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("helvetica", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, T(f"TRUCK VISION AI - Rapport généré automatiquement - Page {self.page_no()}"), align="C")

    def doc_title(self, text: str, sub: str = "") -> None:
        self.set_font("helvetica", "B", 16)
        self.set_text_color(*NAVY)
        self.cell(0, 9, T(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if sub:
            self.set_font("helvetica", "", 10)
            self.set_text_color(*GREY)
            self.cell(0, 6, T(sub), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def section(self, text: str) -> None:
        if self.get_y() > 255:
            self.add_page()
        self.ln(2)
        self.set_font("helvetica", "B", 11)
        self.set_text_color(*NAVY)
        self.cell(0, 7, T(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(210, 215, 225)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(2)

    def kv(self, key: str, value, color=None) -> None:
        self.set_font("helvetica", "", 10)
        self.set_text_color(*GREY)
        self.cell(55, 6.5, T(key))
        self.set_font("helvetica", "B", 10)
        self.set_text_color(*(color or (25, 30, 40)))
        self.multi_cell(0, 6.5, T(value if value not in (None, "") else "-"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def para(self, text: str) -> None:
        self.set_font("helvetica", "", 10)
        self.set_text_color(40, 45, 55)
        self.multi_cell(0, 5.5, T(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def picture(self, relative: str | None, w: float, max_h: float, caption: str = "") -> None:
        if not relative:
            return
        try:
            path = ip.safe_path(relative)
            img = ip.load_image(path)
        except Exception:
            return
        ih, iw = img.shape[:2]
        h = w * ih / iw
        if h > max_h:
            w, h = max_h * iw / ih, max_h
        if self.get_y() + h + 8 > 280:
            self.add_page()
        self.image(str(path), x=15, y=self.get_y(), w=w, h=h)
        self.set_y(self.get_y() + h + 1)
        if caption:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(*GREY)
            self.cell(0, 5, T(caption), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def banner(self, text: str, level: str) -> None:
        self.set_fill_color(*LEVEL.get(level, LEVEL["neutral"]))
        self.set_text_color(255, 255, 255)
        self.set_font("helvetica", "B", 12)
        self.cell(0, 11, T("  Résultat global : " + text), fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)


def _plate_block(pdf: Report, p: dict | None) -> None:
    if not p:
        return
    pdf.kv("Immatriculation", p.get("text") or "Non reconnue")
    pdf.kv("Score de confiance", p.get("confidence"))
    pdf.kv("Statut de reconnaissance", p.get("status"))


def _clean_block(pdf: Report, c: dict | None) -> None:
    if not c:
        return
    pdf.kv("État de propreté", c.get("label"), LEVEL.get(c.get("level", "neutral")))
    pdf.kv("Score de confiance", c.get("confidence"))
    pdf.kv("Statut", c.get("status"))
    if c.get("elements"):
        pdf.kv("Éléments détectés", " ; ".join(c["elements"]))
    if c.get("comment"):
        pdf.kv("Commentaire de l'IA", c["comment"])


def analysis_pdf(d: dict) -> bytes:
    pdf = Report(demo=bool(d.get("demo")))
    pdf.doc_title(f"Rapport de contrôle {d.get('ref', '')}", f"{d.get('date', '')} - {d.get('mode', '')} - {d.get('file_type', '')}")
    pdf.banner(d.get("global_result", "-"), d.get("global_level", "neutral"))

    pdf.section("Informations du contrôle")
    pdf.kv("Identifiant du contrôle", d.get("ref"))
    pdf.kv("Date et heure", d.get("date"))
    pdf.kv("Type d'analyse", d.get("mode"))
    pdf.kv("Fichier analysé", d.get("file_name"))
    pdf.kv("Durée du traitement", d.get("duration"))
    if d.get("checkpoint"):
        pdf.kv("Point de contrôle", d["checkpoint"])
    if d.get("engines"):
        pdf.kv("Moteurs IA", d["engines"])

    files = d.get("files") or {}
    if d.get("file_type") != "Vidéo":
        if d.get("plate"):
            pdf.section("Reconnaissance de plaque")
            _plate_block(pdf, d["plate"])
            pdf.picture(files.get("plate_crop"), 70, 30, "Recadrage de la plaque")
        if d.get("cleanliness"):
            pdf.section("Propreté de la benne")
            _clean_block(pdf, d["cleanliness"])
        pdf.section("Photographie originale")
        pdf.picture(files.get("original"), 180, 110)
    else:
        v = d.get("video") or {}
        pdf.section("Synthèse vidéo")
        pdf.kv("Durée de la vidéo", v.get("duration"))
        pdf.kv("Images analysées", v.get("frames_analyzed"))
        pdf.kv("Camions détectés", v.get("trucks_detected"))
        if v.get("note"):
            pdf.kv("Remarque", v["note"])
        for t in d.get("trucks") or []:
            pdf.section(t.get("title", "Camion"))
            pdf.kv("Présence dans la vidéo", t.get("times"))
            _plate_block(pdf, t.get("plate"))
            _clean_block(pdf, t.get("cleanliness"))
            pdf.picture(t.get("plate_crop"), 60, 26, "Recadrage de la plaque")
            pdf.picture(t.get("snapshot"), 120, 75, "Capture associée")

    if d.get("corrections"):
        pdf.section("Corrections humaines")
        for c in d["corrections"]:
            pdf.para(f"{c.get('date', '')} - {c.get('field', '')} : « {c.get('old', '-')} » -> « {c.get('new', '-')} » ({c.get('author', '')})")
        pdf.ln(1)
        pdf.para("Le résultat original de l'IA est conservé dans l'historique de l'application.")

    pdf.ln(4)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 4.5, T(d.get("disclaimer") or
                             "Les résultats sont produits par un système d'analyse automatique. "
                             "Les scores de confiance indiquent une probabilité et non une certitude ; "
                             "tout résultat marqué « À vérifier » doit être confirmé par un opérateur."))
    return bytes(pdf.output())


def summary_pdf(d: dict) -> bytes:
    pdf = Report(demo=bool(d.get("demo")))
    pdf.doc_title(d.get("title", "Rapport de synthèse"), d.get("period", ""))
    pdf.section("Indicateurs")
    for k, v in d.get("stats") or []:
        pdf.kv(k, v)
    headers, rows, widths = d.get("headers") or [], d.get("rows") or [], d.get("widths") or []
    if headers:
        pdf.section("Détail des contrôles")
        pdf.set_font("helvetica", "B", 8.5)
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(255, 255, 255)
        for h, w in zip(headers, widths):
            pdf.cell(w, 7, T(h), fill=True)
        pdf.ln()
        pdf.set_font("helvetica", "", 8.5)
        for i, row in enumerate(rows):
            pdf.set_fill_color(245, 247, 250)
            pdf.set_text_color(30, 35, 45)
            for v, w in zip(row, widths):
                pdf.cell(w, 6.5, T(v)[: int(w / 1.7)], fill=i % 2 == 0)
            pdf.ln()
    return bytes(pdf.output())
